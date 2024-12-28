from scipy.signal import resample
from scipy.io import wavfile
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import whisper_timestamped as whisper



def mp3_to_wav(input_file, output_file, target_samplerate):
    data, samplerate = sf.read(input_file)
    data = resample(data, int(len(data) * (target_samplerate / samplerate)))
    sf.write(output_file, data, target_samplerate, format='WAV')


def get_path_to_file(filename, subset, audio_format='wav'):
    path = f"data_{audio_format}/{subset}/{filename}.{audio_format}"
    return path


def convert_folder_to_wav(name="train", target_name=None, target_samplerate=16000):
    if target_name is None:
        target_name = name

    # !mkdir "data_wav/{target_name}"
    os.makedirs(f"data_wav/{target_name}", exist_ok=True)

    for file in tqdm(os.listdir(f'data_mp3/{name}'), desc=name + '-->' + target_name):
        filename = file.rsplit('.', 1)[0]
        mp3_to_wav(get_path_to_file(filename, name, 'mp3'),
                   get_path_to_file(filename, target_name, 'wav'),
                   target_samplerate=target_samplerate)


def process_y(y):
    y = y.rename(columns={0: 'audio_name', 1: 'target'})
    y = y.set_index('audio_name')
    return y


def get_word_array(path, start, end, target_samplerate=16000):
    _, data = wavfile.read(path)
    start += 0.1
    end += 0.25

    data = data[int(target_samplerate * start):int(target_samplerate * end)]
    return data


def get_subset(data_type):
    if data_type.endswith('train'):
        return 'train'
    elif data_type.endswith('test'):
        return 'test'
    else:
        raise ValueError()


def process_whisper_res(whisper_res, y, target_letters, letters, threshold=0.5):
    """
    threshold -  Порог для confidence виспера
    """

    np.random.seed(42)

    data = []

    scale_factor = 2 ** 15  # На что делим при переводе аудио из int во float

    for filename in tqdm(whisper_res.keys()):
        label, data_type = y.loc[filename, ['target', 'data_type']].values
        subset = get_subset(data_type)

        for segment in whisper_res[filename]['segments']:
            words = segment['words']
            for word in words:
                try:
                    text = word['text'].lower()
                except:
                    text = word['word'].lower()
                text = ''.join(letter for letter in text if letter in letters)
                if len(text) <= 2:  # Слишком короткий текст
                    continue

                start = word['start']
                end = word['end']
                try:
                    confidence = word['confidence']
                except:
                    confidence = word['probability']
                if confidence < threshold: continue  # Низкий confidence виспера

                d = {'file': filename, 'text': text, 'start': start, 'end': end,
                     'confidence': confidence, 'label': label,
                     'data_type': data_type, 'subset': subset}

                for letter in target_letters:
                    count = text.count(letter)
                    d[f"{letter}_count"] = count

                data.append(d)

    data = pd.DataFrame(data)
    data['path'] = data[['file', 'subset']].apply(lambda x: f'data_wav/{x["subset"]}/' + x['file'][:-4] + '.wav',
                                                  axis=1)

    arrays = []

    drop_ids = []

    for idx, (path, start, end) in tqdm(data[['path', 'start', 'end']].iterrows(), total=len(data)):
        array = get_word_array(path, start, end) / scale_factor
        if len(array) == 0:  # Пустой отрезок вырезали
            drop_ids.append(idx)
            continue

        arrays.append(array)

    data = data.drop(index=drop_ids)

    return data, arrays


def process_audio_folder_by_whisper(path):
    model = whisper.load_model("large", device="cuda")

    res_whisper = {}
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        # print(full_path, os.path.exists(full_path))
        audio = whisper.load_audio(full_path)
        result = whisper.transcribe(model, audio, language="ru")
        res_whisper[name] = result
    return res_whisper
