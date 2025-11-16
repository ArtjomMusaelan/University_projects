import whisper
import numpy
import pandas as pd
import re

model = whisper.load_model("turbo")

result = model.transcribe("task_2/40_minutes_clipped.mp3", language="ru")

text = result["text"]

parsed_text = [[sentence.strip()] for sentence in re.split(r'(?<=[!?.])\s+', text)][:-1]

df = pd.DataFrame(parsed_text, columns=["Sentence"])

df.to_csv("task_2/transcribed_data_whisper.csv", sep=";", index=False)