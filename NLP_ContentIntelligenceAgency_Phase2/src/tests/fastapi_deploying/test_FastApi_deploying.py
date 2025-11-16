import io
import textwrap

# --- General/Health checks ---


def test_root_and_health(fastapi_client):
    r1 = fastapi_client.get("/")
    r2 = fastapi_client.get("/health")
    assert r1.status_code == 200 and "running" in r1.json()["message"].lower()
    assert r2.status_code == 200 and r2.json() == {"status": "healthy"}


# --- /predict endpoint ---


def test_predict_valid(fastapi_client):
    resp = fastapi_client.post("/predict", json={"text": "I am happy"})
    js = resp.json()
    assert resp.status_code == 200
    assert set(js) >= {"text", "predicted_label", "emotions"}
    assert len(js["emotions"]) == 7


def test_predict_invalid(fastapi_client):
    # Missing 'text' field
    resp = fastapi_client.post("/predict", json={"sentence": "I am sad"})
    assert resp.status_code in (400, 422)


def test_predict_empty(fastapi_client):
    resp = fastapi_client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    js = resp.json()
    assert set(js) >= {"text", "predicted_label", "emotions"}


def test_predict_out_of_distribution(fastapi_client):
    resp = fastapi_client.post("/predict", json={"text": "jkfdslajkf"})
    assert resp.status_code == 200
    js = resp.json()
    assert set(js) >= {"text", "predicted_label", "emotions"}


# --- /upload-csv endpoint ---


def test_upload_csv_valid(fastapi_client):
    csv_content = textwrap.dedent("""\
        text
        Hello world
        I am tired
    """)
    csv_bytes = io.BytesIO(csv_content.encode())
    resp = fastapi_client.post(
        "/upload-csv",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    out = resp.json()
    assert "file_path" in out and out["file_path"].endswith("csv_predictions.csv")


def test_upload_csv_invalid(fastapi_client):
    # CSV missing 'text' column
    bad_csv = io.BytesIO(b"sentence\nWrong header")
    resp = fastapi_client.post(
        "/upload-csv",
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert resp.status_code >= 400 or "error" in resp.text.lower()


# --- /feedback-csv endpoint ---


def test_feedback_csv_valid(fastapi_client):
    csv_content = "text,predicted_label\nIt works!,happy"
    csv_bytes = io.BytesIO(csv_content.encode())
    resp = fastapi_client.post(
        "/feedback-csv",
        data={"feedback": "good"},
        files={"csv_file": ("good.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    out = resp.json()
    assert "uploaded" in out["status"]


def test_feedback_csv_invalid(fastapi_client):
    # Missing CSV file
    resp = fastapi_client.post("/feedback-csv", data={"feedback": "good"})
    assert resp.status_code >= 400 or "error" in resp.text.lower()


# --- /youtube-transcript endpoint ---


def test_youtube_stub(fastapi_client, monkeypatch):
    # Mocks to avoid network/real yt-dlp/whisper calls in test
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    import nltk
    monkeypatch.setattr(nltk, "sent_tokenize", lambda txt: ["first sentence", "second"])
    resp = fastapi_client.post(
        "/youtube-transcript",
        params={"url": "https://youtu.be/dQw4w9WgXcQ"},
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["file_path"].endswith("youtube_transcript.csv")


# --- /upload-audio-transcript endpoint ---


def test_audio_transcript_stub(fastapi_client, monkeypatch):
    # Mocks to avoid real Whisper call
    import nltk
    monkeypatch.setattr(
        nltk, "sent_tokenize", lambda txt: ["audio sentence one", "audio sentence two"]
    )

    class DummyWhisper:
        def transcribe(self, f):
            return {"text": "audio sentence one. audio sentence two"}

    import sys
    sys.modules["whisper"] = __import__("types")
    setattr(sys.modules["whisper"], "load_model", lambda _: DummyWhisper())

    audio_bytes = io.BytesIO(b"FAKEAUDIO")
    resp = fastapi_client.post(
        "/upload-audio-transcript",
        files={"file": ("audio.mp3", audio_bytes, "audio/mp3")},
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["file_path"].endswith("uploaded_audio_transcript.csv")
