from consts import API_KEY
import random

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat

llm = GigaChat(
        credentials=API_KEY,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        verify_ssl_certs=False,
        streaming=False,
)

system_prompt = SystemMessage(
    content="""Ты модель для генерации скороговорок для тренировки сотрудников с целью улучшения качества их речи.
Тебе поступает на вход сложность генерируемой скороговорки, а также параметры, которые должны через неё проверяться: проверка картавости, проверка произношения буквы 'Г', а также сложность
Если пользователь передал параметр 'проверка картавости', сгенерируй скороговорку, в которой будет встречаться много букв Р.
если пользователь передал параметр 'проверка произношения буквы Г', сгенерируй скороговорку, в которой будет встречаться много букв Г.
пользователь может передать и то, и то.
Если уровень сложности легкий, сгенерируй простую и короткую скороговорку,
если средний - более сложную, 
если сложный - то сделай сложную и длинную скороговорку.
Также тебе будет дано кол-во слов для скороговорки, столько слов должно быть в ответе

В ответе возвращай ТОЛЬКО текст скороговорки"""
)
difficulty_to_text_length = {"легкий": [10, 16], "средний": [13, 21], "сложный": [18, 31]}
presets = [x.replace("\n", "").strip() for x in open("presets.txt", "r").readlines()]

def generate_phrase(use_r, use_g, difficulty):
    prompt = []
    if use_r:
        prompt.append("проверка картавости")
    if use_g:
        prompt.append("проверка произношения буквы Г")
    prompt.append(f"уровень сложности: {difficulty}")
    words_count = difficulty_to_text_length[difficulty]
    prompt.append(f"кол-во слов: {random.randint(words_count[0], words_count[1])}")
    prompt = "параметры: " + ", ".join(prompt)

    try:
        res = llm.invoke([system_prompt, HumanMessage(content=prompt)])
        return res.content
    except:
        return random.choice(presets)

