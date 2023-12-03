from langchain.llms import Ollama
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import Audio, display
from TTS.api import TTS
import sounddevice as sd

from multiprocessing import Process, Manager

#m = Manager()
#q = m.Queue()
#p = Process(target = gettext, args = (bot,message,thread_history,users[message.from_user.id].lang,q)).start()
#q.put(history)
#
#
#
#    try:
#        A = q.get(False)
#    except: 
#        pass

with torch.no_grad():
    ollama = Ollama(base_url='http://localhost:11434',
    model="mistral")
#with torch.no_grad():
#    ollama = Ollama(base_url='http://localhost:11434',
#    model="llama2:13b")

print(TTS().list_models())

prompt = 'What is your age? Give a short answer'
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph").to("cuda")

#tts = TTS("tts_models/multilingual/multi-dataset/bark").to("cuda")
#tts = TTS("tts_models/en/multi-dataset/tortoise-v2").to("cuda")
text_chunk = []

for chunk in ollama._stream(prompt):
    text_chunk.append(chunk.text)
    if ('.' in chunk.text) or (',' in chunk.text) or (';' in chunk.text):
        print(''.join(text_chunk))

        gen = tts.tts(''.join(text_chunk))
        #gen = tts.tts(''.join(text_chunk), language="en", speaker = "lj",preset = "ultra_fast")
        sd.wait(ignore_errors=True)
        sd.play(gen,24000)#,blocking =True
        text_chunk = []
    
sd.wait(ignore_errors=True)
