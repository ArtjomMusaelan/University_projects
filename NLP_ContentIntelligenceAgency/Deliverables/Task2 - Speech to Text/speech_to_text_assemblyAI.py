import assemblyai as aai
import numpy as np
import pandas as pd
import re

aai.settings.api_key = "<API_KEY>"

ru_config = aai.TranscriptionConfig(language_code="ru")

transcriber = aai.Transcriber(config=ru_config)

transcript = transcriber.transcribe("task_2/40_minutes_clipped.mp3")

text = transcript.text

print(text)

parsed_text = [[sentence.strip()] for sentence in re.split(r'(?<=[!?.])\s+', text)][:-1]

df = pd.DataFrame(parsed_text, columns=["Sentence"])

df.to_csv("task_2/transcribed_data_assemblyAI.csv", sep=";", index=False)

