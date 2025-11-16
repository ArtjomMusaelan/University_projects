import logging
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import whisper
import tempfile
import subprocess
import io
import os
import csv
import nltk
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication

# ------------------------ Setup Logging ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------ Azure Endpoint Config (use env vars) ------------------
# Replace these environment variables with your actual values in deployment.
scoring_uri = os.environ.get("SCORING_URI", "https://<your-azure-endpoint>/score")
api_key = os.environ.get("AZUREML_MODEL_KEY", "<your-model-key>")
headers = {
    "Content-Type": "application/json",
    "azureml-model-key": api_key
}

# ------------------ Feedback CSV Upload (use env vars) ------------------
def upload_feedback_csv_to_azureml(feedback_type="good", local_file="csv_predictions.csv"):
    # Use environment variables for all sensitive information
    tenant_id = os.environ.get("AZURE_TENANT_ID", "<your-tenant-id>")
    service_principal_id = os.environ.get("AZURE_SP_ID", "<your-service-principal-id>")
    service_principal_password = os.environ.get("AZURE_SP_PASSWORD", "<your-service-principal-password>")
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "<your-resource-group>")
    workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "<your-workspace-name>")
    datastore_name = os.environ.get("AZURE_DATASTORE_NAME", "workspaceblobstore")

    auth = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=service_principal_id,
        service_principal_password=service_principal_password
    )
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=auth
    )
    datastore = Datastore.get(ws, datastore_name)
    target_path = f"feedback/{feedback_type}/"
    remote_path = os.path.join(target_path, os.path.basename(local_file))
    datastore.upload_files(
        files=[local_file],
        target_path=target_path,
        overwrite=True,
        show_progress=True
    )
    Dataset.File.from_files((datastore, remote_path)).register(
        workspace=ws,
        name=f"feedback_{feedback_type}_csv",
        description=f"{feedback_type.capitalize()} feedback CSV from user submission.",
        create_new_version=True
    )

# ----------------------- Input Schema -------------------------
class TextInput(BaseModel):
    text: str

# ---------------------- Initialize App ------------------------
app = FastAPI()
logger.info("Emotion Classification API (via Azure) initialized.")

@app.get("/")
async def root():
    return {"message": "Emotion Classification API (Azure endpoint) is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ------------------------ Predict Endpoint ---------------------------
@app.post("/predict")
async def predict_emotion(input_text: TextInput):
    logger.info("Sending prediction to Azure endpoint for: %s", input_text.text)
    try:
        response = requests.post(
            scoring_uri,
            headers=headers,
            json={"text": input_text.text},
            timeout=10
        )
        response.raise_for_status()
        prediction = response.json()
        prediction["text"] = input_text.text
        logger.info("Prediction received: %s", prediction)
        return prediction
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return {"error": "Prediction failed.", "detail": str(e)}

# -------------------- Upload CSV --------------------------
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    logger.info("CSV upload request received: %s", file.filename)
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error("Error reading CSV: %s", str(e))
        return {"error": "Failed to read CSV file."}

    if "text" not in df.columns:
        return {"error": "CSV must contain a 'text' column."}

    df = df[["text"]]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "predicted_emotion"])

    for sentence in df["text"]:
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            response = requests.post(
                scoring_uri,
                headers=headers,
                json={"text": sentence},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            label = result.get("predicted_label", "unknown")
            writer.writerow([sentence, label])
        except Exception as e:
            logger.error("Failed prediction for: %s | %s", sentence, str(e))
            writer.writerow([sentence, "error"])

    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "csv_predictions.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(output.getvalue())

    return {"message": "CSV processed.", "file_path": output_path}

# ------------------- Feedback CSV Upload ----------------------
@app.post("/feedback-csv")
async def feedback_csv(
    feedback: str = Form(...),
    csv_file: UploadFile = File(...)
):
    local_path = f"/tmp/{csv_file.filename}"
    with open(local_path, "wb") as f:
        f.write(await csv_file.read())
    try:
        upload_feedback_csv_to_azureml(feedback_type=feedback, local_file=local_path)
    except Exception as e:
        logger.error("Failed to upload feedback CSV: %s", e)
        return {"status": "error", "detail": str(e)}
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    return {"status": "uploaded", "feedback": feedback}

# ------------------- Download CSV ----------------------
@app.get("/download-csv")
async def download_csv():
    path = "/app/output/csv_predictions.csv"
    if os.path.exists(path):
        return FileResponse(path, filename="csv_predictions.csv", media_type="text/csv")
    return {"error": "CSV file not found."}

# ------------------- Transcript Endpoints ----------------------
nltk.download("punkt")

@app.post("/youtube-transcript")
async def generate_transcript_csv(url: str):
    logger.info("Received YouTube URL for transcript: %s", url)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", audio_path, url
            ], check=True)
            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(audio_path)
            transcript = result["text"]
    except Exception as e:
        logger.error("Transcription failed: %s", str(e))
        return {"error": "Transcription failed."}

    sentences = nltk.sent_tokenize(transcript)
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["text"])
    for s in sentences:
        writer.writerow([s.strip()])

    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "youtube_transcript.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())
    return {"message": "Transcript CSV saved.", "file_path": output_path}

@app.post("/upload-audio-transcript")
async def generate_transcript_from_audio(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, file.filename)
            with open(input_path, "wb") as f:
                f.write(await file.read())
            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(input_path)
            transcript = result["text"]
    except Exception as e:
        return {"error": str(e)}

    sentences = nltk.sent_tokenize(transcript)
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["text"])
    for s in sentences:
        writer.writerow([s.strip()])

    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uploaded_audio_transcript.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())
    return {"message": "Transcript CSV saved.", "file_path": output_path}
