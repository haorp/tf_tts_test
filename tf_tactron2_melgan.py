import numpy as np
import soundfile as sf
import yaml
import IPython.display as ipd
import tensorflow as tf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

# initialize tactron2 model.
fs_config = AutoConfig.from_pretrained('./examples/tacotron2/conf/tacotron2.baker.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=fs_config,
    pretrained_path="../tactron2_melgan_model/model-200000.h5"
)

# initialize multiband_melgan model
melgan_config = AutoConfig.from_pretrained('./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
    pretrained_path="../tactron2_melgan_model/generator-200000.h5"
)

# inference
processor = AutoProcessor.from_pretrained(pretrained_path="./test/files/baker_mapper.json")

#tts synthesis
def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text,True)

  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

  # vocoder part
  if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  elif vocoder_name == "MB-MELGAN":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
  else:
    return mel_outputs.numpy(), audio.numpy()

input_text="你好，我是客服小好，有什么需要帮忙的"

mels, alignment_history, audios = do_synthesis(input_text, tacotron2, melgan, "TACOTRON", "MB-MELGAN")

# save to file
ipd.Audio(audios, rate=22050)
sf.write('./audio/audio_after.wav', audios, 22050, "PCM_16")