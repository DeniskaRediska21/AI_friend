
from langchain.llms import Ollama
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import sounddevice as sd

from bark import SAMPLE_RATE, generate_audio, preload_models
import os



preload_models(
    text_use_small=True,
    coarse_use_small=True,
    fine_use_gpu=False,
    fine_use_small=True,
)

with torch.no_grad():
    ollama = Ollama(base_url='http://localhost:11434',
    model="orca-mini")


prompt = 'why is the sky blue?'

#tts = TTS("tts_models/multilingual/multi-dataset/bark").to("cuda")
#tts = TTS("tts_models/en/multi-dataset/tortoise-v2").to("cuda")
text_chunk = []

for chunk in ollama._stream(prompt):
    text_chunk.append(chunk.text)
    if ('.' in chunk.text) or (',' in chunk.text) or (';' in chunk.text):
        print(''.join(text_chunk))

        gen = generate_audio(''.join(text_chunk))

#, history_prompt="v2/en_speaker_1")
        
        #gen = tts.tts(''.join(text_chunk), language="en", speaker = "lj",preset = "ultra_fast")
        sd.wait(ignore_errors=True)
        sd.play(gen,SAMPLE_RATE)#,blocking =True
        text_chunk = []
    
sd.wait(ignore_errors=True)
