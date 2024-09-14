from utils.generation import SAMPLE_RATE, generate_audio, preload_models, audio_rec
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
Hello, my name is Nose. And uh, and I like hamburger. Hahaha... But I also have other interests such as playing tactic toast.
"""
# audio_array = generate_audio(text_prompt)
# audio_prompt = '/home/xintong/VALL-E-X/presets/librispeech_4.npz'
# audio_prompt = '/data2/xintong/tts/LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav'
audio_prompt = 'output/vallex_generation.wav'
audio_array = audio_rec(prompt=audio_prompt)
# save audio to disk
write_wav("output/rec/vallex_generation_demo1.wav", SAMPLE_RATE, audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)