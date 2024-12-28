import requests
import json
import soundfile as sf


def get_audio_score(audio_path):
    with open(audio_path, "rb") as file:
        response = requests.post(
            "https://tbankspeech.ru/api/predict",
            files={"file": (audio_path.split("/")[-1], file, "audio/mpeg")}
        )

    try:
        none_score, r_score, g_score, stut_score = json.loads(response.content)["predict_proba"]
        print(none_score, r_score, g_score, stut_score)
        if r_score > 0.2:
            r_score *= min(1, r_score * 1.5)
        else:
            r_score = r_score * 0.8
        g_score = min(1, g_score * 1.4)

        total_score = min((r_score + g_score + stut_score) / 1.2, 1)

        audio, sample_rate = sf.read(audio_path)
        speed_score = (len(audio) / sample_rate) / (sample_rate * 3)
    except:
        r_score, g_score, stut_score, speed_score, total_score = 0.25, 0.1, 0.05, 0.8, 0.125

    return {"r_score": round(r_score * 100, 2), "g_score": round(g_score * 100, 2),
            "stut_score": round(stut_score * 100, 2),
            "speed_score": round(speed_score * 100, 2), "total_score": round(total_score * 100, 2)}