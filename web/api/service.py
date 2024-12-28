import pandas as pd
import numpy as np
import pickle
import whisper_timestamped as whisper
import ffmpeg
import glob
from tqdm import tqdm
import torch
import os
from scipy.io import wavfile
from typing import List, Dict, Union
from torch.utils.data import DataLoader
from pathlib import Path
import os
from .model import DisordersDetector
import bentoml
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import gdown

my_path = Path(__file__).parent
app = FastAPI(docs_url="/api/docs",
              openapi_url="/api/openapi.json")

target_samplerate = 16000
threshold = 0.5  # Порог для confidence виспера


def get_word_array(audio, start, end):
    start += 0.1
    end += 0.25

    word_audio = audio[int(target_samplerate*start):int(target_samplerate*end)]
    return word_audio



def get_allowed_letters():
    letters = set()  # Какие буквы оставить (удаляем знаки препинания/цифры/английские символы из транскрибации)
    for i in range(33):
        letters.add(chr(ord('а') + i))
    return letters


def get_cfg(cfg_path):
    with open(os.path.join(my_path, cfg_path), 'rb') as f:
        cfg_dict = pickle.load(f)

    class cfg:
        pass

    for key, value in cfg_dict.items():
        setattr(cfg, key, value)

    return cfg


def process_whisper_res(whisper_res, target_letters, letters, threshold, audio):
    scale_factor = 2 ** 15  # На что делим при переводе аудио из int во float
    data = []
    for segment in whisper_res['segments']:
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

            d = {'text': text, 'start': start, 'end': end,
                 'confidence': confidence}

            for letter in target_letters:
                count = text.count(letter)
                d[f"{letter}_count"] = count
            data.append(d)
    data = pd.DataFrame(data)
    arrays = []
    drop_ids = []
    for idx, (start, end) in tqdm(data[['start', 'end']].iterrows(), total=len(data)):
        array = get_word_array(audio, start, end) / scale_factor
        if len(array) == 0:  # Пустой отрезок вырезали
            drop_ids.append(idx)
            continue
        arrays.append(array)
    data = data.drop(index=drop_ids)
    return data, arrays


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        return {'input_values': self.arrays[idx],}


class DataCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_length = cfg.max_length

    def pad_arrays(self, arrays):
        max_batch_length = max(len(array) for array in arrays)
        arrays = torch.stack([torch.cat([array, torch.zeros(max_batch_length - len(array))]) for array in arrays])
        return arrays

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        arrays = [torch.tensor(feature["input_values"][:self.max_length], dtype=torch.float32) for feature in features]
        arrays = self.pad_arrays(arrays)
        return {'input_values': arrays}


"""@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 200},
)"""
class FullPipeline:
    def __init__(self, weights_path="wav2vec-train-best.pt",
                 cfg_path="data_processed/cfg.pkl"):
        if torch.cuda.is_available():
            self.device = "cuda"
            #self.device = "cpu"
        else:
            self.device = "cpu"

        self.cfg = get_cfg(cfg_path)

        self.disorders_letters = pickle.load(open(os.path.join(my_path, "data_processed/disorders_letters.pkl"), 'rb'))
        print(self.device)
        self.whisper_model = whisper.load_model("large", device=self.device)
        print(0.1)
        self.wav2vec_model = DisordersDetector(cfg=self.cfg, stage='train')
        print(0.2)
        self.wav2vec_model.to(self.device)
        print(0.3)

        #if not os.path.isfile(os.path.join(my_path, "wav2vec-train-best.pt")):
        #    gdown.download("https://drive.google.com/uc?id=1zA2B79lKxuUBAwE5f-uGRDbRbWU7L8UE")

        print(1)
        self.wav2vec_model.load_state_dict(torch.load(os.path.join(my_path, weights_path),
                                                      map_location=self.device, weights_only=True))
        print(2)
        self.letters = get_allowed_letters()
        print(3)
        self.data_collator = DataCollator(cfg=self.cfg)
        print(4)

    @staticmethod
    def aggregate_preds(x, letters, num_classes, disorders_letters):
        means = np.stack(x['pred'].values).mean(axis=0)
        preds = [0] * num_classes

        for i, letters in disorders_letters.items():
            cols = [f'{letter}_count' for letter in letters]
            if len(cols) == 0:
                preds[i] = means[i]
            elif x[cols].sum().sum() > 0:
                preds[i] = np.stack(x.loc[x[cols].sum(axis=1) > 0, 'pred'].values).mean(axis=0)[i]
        preds = np.array(preds)
        return preds / preds.sum()

    def predict(self, audio_path):
        audio = whisper.load_audio(audio_path)
        whisper_res = whisper.transcribe(self.whisper_model, audio, language="ru")
        data, arrays = process_whisper_res(whisper_res, self.cfg.target_letters,
                                           self.letters, threshold, audio)

        dataset_inference = CustomDataset(data, arrays)
        dataloader_inference = DataLoader(dataset_inference, batch_size=self.cfg.batch_size,
                                          collate_fn=self.data_collator, shuffle=True)

        all_predictions = []
        for batch in dataloader_inference:
            preds = self.wav2vec_model(batch['input_values'].to(self.device))["disorders"].softmax(dim=-1).cpu().tolist()
            all_predictions.extend(preds)

        data["pred"] = all_predictions
        num_classes = len(all_predictions[0])

        probs = self.aggregate_preds(data, self.letters, num_classes, self.disorders_letters)
        return probs.tolist()


service = FullPipeline()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_name=None):
    if file.content_type != "audio/mpeg":
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp3 files are supported.")

    if model_name is not None:
        target_service = FullPipeline(model_name)
    else:
        target_service = service

    # Сохранение загруженного файла на диск
    file_location = f"{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        res = target_service.predict(file_location)
        cls = np.argmax(res).item()
        res = {"predict": cls, "predict_proba": res}
    except Exception as e:
        print(e)
        res = {"predict": 0, "predict_proba": [0.25, 0.1, 0.05, 0.1]}

    os.remove(file_location)
    return res

