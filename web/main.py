import pandas as pd
import streamlit as st
from authentification import *
import streamlit_antd_components as sac
from st_audiorec_ru import st_audiorec
from texts import *
from phrase_generator import generate_phrase
from db_api import *
from components import *
from model_api import get_audio_score
import plotly.express as px
from annotated_text import annotated_text
from streamlit_searchbox import st_searchbox
import numpy as np
from components import Searcher
import streamlit.components.v1 as components
import zipfile
import os
import tempfile
import soundfile as sf

import time
import random
import hashlib
import requests
import datetime
import json


if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state, layout="wide")

if "start_register" in st.session_state and st.session_state.start_register:
    signup()
#elif "login" not in st.session_state:
#    signin()
#st.session_state["login"] = "test"

with st.sidebar:
    col1, col2 = st.columns((4, 7))
    if not ("login" not in st.session_state or st.session_state["login"] is None):
        col2.write(f"Здравствуйте, {st.session_state.login}")
        if col1.button("Выйти"):
            st.session_state.login = None
    else:
        if col1.button("Войти"):
            signin()

    pages_tree = sac.menu(items=[
        sac.MenuItem("Тренажёр", children=[sac.MenuItem("Тренировка"),
                                           sac.MenuItem("Мой прогресс"),
                                           sac.MenuItem("История")], icon=sac.AntIcon("AudioOutlined")),
        sac.MenuItem("Аналитика диалогов", children=[sac.MenuItem("Загрузить набор аудио"),
                                                     sac.MenuItem("Аналитика")], icon=sac.AntIcon("BarChartOutlined")),
        sac.MenuItem("О нас", children=[sac.MenuItem("Наши алгоритмы"),
                                        sac.MenuItem("API"),
                                        sac.MenuItem("Команда")], icon=sac.AntIcon("TeamOutlined"))
                          ], open_all=True)

if pages_tree == "Тренажёр":
    pages_tree = "Тренировка"

if pages_tree == "Тренировка":
    st.markdown("## Тренировка")
    st.write(TRAINING_INTRO_TEXT)
    st.write(TRAINING_GUIDE_TEXT)

    # Встраиваем HTML/JavaScript для запроса разрешения на микрофон
    mic_permission = """
        <script>
        async function requestMic() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log('Доступ к микрофону предоставлен');
                return true;
            } catch (err) {
                console.error('Ошибка при запросе доступа к микрофону:', err);
                return false;
            }
        }

        const isGranted = await requestMic();
        if (isGranted) {
            Streamlit.setComponentValue('Доступ предоставлен');
        } else {
            Streamlit.setComponentValue('Доступ отклонен');
        }
        </script>
    """

    # Встраивание компонента в Streamlit
    mic_status = components.html(mic_permission, height=0)

    col1, col2 = st.columns((5, 7))
    with col1.popover(label="Настроить уровень сложности", disabled=False, use_container_width=True, ):
        if "selected_r_setting" not in st.session_state:
            st.session_state.selected_r_setting = True
            st.session_state.selected_g_setting = True
            st.session_state.use_stuttering = True
            st.session_state.selected_diff_setting = 0

        use_r = sac.switch(label='Проверка картавости', align='left', size='md',
                           value=st.session_state.selected_r_setting)
        use_g = sac.switch(label='Проверка произношения буквы "Г"', align="left", size="md",
                           value=st.session_state.selected_g_setting)
        use_stuttering = sac.switch(label="Проверка заикания", align="left", size="md",
                                    value=st.session_state.use_stuttering)
        difficulty = sac.segmented(items=[sac.SegmentedItem(label="легкий"),
                                          sac.SegmentedItem(label="средний"),
                                          sac.SegmentedItem(label="сложный")], label="Выберите уровень сложности",
                                   index=st.session_state.selected_diff_setting)
        st.session_state.selected_r_setting = use_r
        st.session_state.selected_g_setting = use_g
        st.session_state.use_stuttering = use_stuttering
        st.session_state.selected_diff_setting = {"легкий": 0, "средний": 1, "сложный": 2}[difficulty]

    if "but_text" not in st.session_state:
        st.session_state.but_text = "Приступить к тренировке"

    if st.button("Приступить к тренировке" if "last_audio" not in st.session_state else "Повторить тренировку"):
        st.session_state.start_training = True
        st.session_state.show_full_text = False
        st.session_state.show_audio_record_widget = True
        st.session_state.last_audio = None
        st.session_state.generated_text = generate_phrase(use_r, use_g, difficulty)
        st.rerun()

    if "start_training" in st.session_state and st.session_state.start_training:
        st.write(":red[Обратите внимание: если длина аудиозаписи равна нулю, проблема может быть в устройстве звукозаписи, рекомендуется переключиться на динамики устройства.]")
        st.markdown("##### Нажмите на запись и прочитайте этот текст: ")
        generated_text = st.session_state.generated_text
        text_field = st.empty()
        if not st.session_state.show_full_text:
            text = ""
            while len(generated_text) > 0:
                s = random.randint(1, 3)
                part = generated_text[:s]
                text += part
                generated_text = generated_text[s:]
                text_field.write(text)
                time.sleep(0.05)
            st.session_state.show_full_text = True
        else:
            text_field.write(generated_text)

        if st.session_state.show_audio_record_widget:
            wav_audio_data = st_audiorec()

            if wav_audio_data is not None:
                created_at = str(datetime.datetime.now())
                hash_value = int(hashlib.md5(created_at.encode()).hexdigest(), 16) % (2 ** 32)
                audio_path = f"audios/{hash_value}.mp3"
                with open(audio_path, "wb") as f:
                    f.write(wav_audio_data)
                data, samplerate = sf.read(audio_path)
                sf.write(audio_path, data[:, 0], samplerate, format='MP3')

                if "login" in st.session_state and st.session_state.login is not None:
                    add_user_audio(st.session_state.login, hash_value, created_at)
                else:
                    st.write(":red[Для сохранения вашей аудиозаписи в историю, войдите в аккаунт]")

                st.session_state.show_audio_record_widget = False
                st.session_state.last_audio = wav_audio_data
                st.session_state.last_scores = get_audio_score(audio_path)
                if "login" in st.session_state and st.session_state.login is not None:
                    add_score(hash_value, st.session_state.last_scores["r_score"],
                              st.session_state.last_scores["g_score"], st.session_state.last_scores["stut_score"],
                              st.session_state.last_scores["speed_score"],
                              st.session_state.last_scores["total_score"])
        else:
            col1, col2 = st.columns((4, 6))
            col1.audio(st.session_state.last_audio)

        if "last_scores" in st.session_state:
            scores = st.session_state.last_scores
            scores_desription(scores, indents_count=59)



if pages_tree == "История":
    if not ("login" in st.session_state and st.session_state.login is not None):
        st.write(f":red[Для того, чтобы посмотреть историю тренировок, войдите в аккаунт.]")
    else:
        user_audios = get_audio_hashes_by_login(st.session_state.login)

        st.session_state.selected_page = sac.pagination(align='center', jump=True, show_total=True,
                                                        total=len(user_audios), page_size=5)

        pg = st.session_state.selected_page
        user_audios_page = user_audios[(pg-1)*5:pg*5]
        but_clicks = []
        for k, audio in enumerate(user_audios_page):
            cont = st.container(border=True)
            col1, col2, col3 = cont.columns((2, 12, 1))
            col1.write(audio[1].split(".")[0])
            with col2:
                scores = get_score_by_audio_hash(audio[0])
                cols_scores = col2.columns((1, 1, 1, 1, 1))
                for i, (sc, sc_ru) in enumerate([("r_score", "Буква Р"), ("g_score", 'Гэканье'), ("stut_score", "Заикание"),
                                  ("speed_score", "Скорость чтения"), ("total_score", "Общий рейтинг")]):
                    if scores[sc] > 75:
                        color = "rgb(245, 108, 108)"
                    elif scores[sc] > 45:
                        color = "rgb(240, 240, 100)"
                    else:
                        color = "rgb(108, 245, 137)"
                    with cols_scores[i]:
                        annotated_text((sc_ru, f"{scores[sc]}%", color))

            but_clicks.append(col3.button(label="", icon="🗑️", key=k))
            with cont.popover(label="Посмотреть детали", use_container_width=True):
                st.audio(f"audios/{audio[0]}.mp3")
                scores_desription(scores, indents_count=54, key=k)
        for i in range(len(but_clicks)):
            if but_clicks[i]:
                delete_audio_by_hash_and_login(user_audios[i][0], st.session_state.login)
                st.rerun()

if pages_tree == "Мой прогресс":
    if not ("login" in st.session_state and st.session_state.login is not None):
        st.write(f":red[Для того, чтобы посмотреть историю тренировок, войдите в аккаунт.]")
    else:
        user_audios = get_audio_hashes_by_login(st.session_state.login)
        if len(user_audios) < 5:
            st.write(":red[Чтобы получить подробную аналитику вашего прогресса, пройдите минимум 5 тренировок]")
        else:
            col1, col2 = st.columns((3, 7))
            with col1:
                fig = px.line(x=[x[1] for x in user_audios], y=[x[0] for x in user_audios],
                              title="История тренировок")
                fig.update_layout(
                    xaxis_title='Дата',
                    yaxis_title='Кол-во тренировок'
                )
                st.plotly_chart(fig, use_container_width=True)
                scores = [get_score_by_audio_hash(x[0]) for x in user_audios]

            with col2:
                history_df = {
                    "x": sum([[u[1] for u in user_audios] for _ in range(5)], []),
                    "y": sum([[x[s] for x in scores] for s in ["r_score", "g_score", "stut_score", "speed_score", "total_score"]], []),
                    "category": sum([[s for _ in range(len(user_audios))] for s in ["картавость", '"ГЭканье"', "Заикание",
                                                                                    "скорость чтения", "общий рейтинг"]], [])
                }
                history_df = pd.DataFrame(history_df)
                fig = px.line(history_df, x="x", y="y", color="category", title="Динамика прогресса")
                fig.update_layout(
                    xaxis_title='Дата тренировки',
                    yaxis_title='Рейтинг'
                )
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns((6, 4))
            with col1:
                sorted_scores = [x[0] for x in list(sorted([(scores[i],
                                              datetime.datetime.strptime(user_audios[i][1], "%Y-%m-%d %H:%M:%S.%f"))
                                             for i in range(len(user_audios))],
                                            key=lambda x: x[1]))]

                data = {
                    'Уровень картавости': [0, np.mean([x["r_score"] for x in sorted_scores[-5:]])],
                    'Ошибки в буквах "Г"': [1, np.mean([x["g_score"] for x in sorted_scores[-5:]])],
                    "Заикание": [2, np.mean([x["stut_score"] for x in sorted_scores[-5:]])],
                    'Скорость чтения': [3, np.mean([x["speed_score"] for x in sorted_scores[-5:]])],
                    'Общий рейтинг': [4, np.mean([x["total_score"] for x in sorted_scores[-5:]])]
                }
                df = pd.DataFrame(data).transpose()
                fig = px.bar(df, x=0, y=1, color=1, color_continuous_scale=generate_gradient(100),
                             title="Уровень ошибок за последние тренировки")
                fig.update_layout(title="Распределение ошибок в последних аудио",
                                  xaxis_title="Картавость    'ГЭканье'     Картавость     Скорость чтения     Общий рейтинг",
                                  yaxis_title="Уровень ошибки")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                r_diff = round(np.mean(np.diff([x["r_score"] for x in sorted_scores[-5:]])), 2)
                g_diff = round(np.mean(np.diff([x["g_score"] for x in sorted_scores[-5:]])), 2)
                stut_diff = round(np.mean(np.diff([x["stut_score"] for x in sorted_scores[-5:]])), 2)
                speed_diff = round(np.mean(np.diff([x["speed_score"] for x in sorted_scores[-5:]])), 2)
                total_diff = round(np.mean(np.diff([x["total_score"] for x in sorted_scores[-5:]])), 2)

                show_progress_text(data, r_diff, g_diff, stut_diff, speed_diff, total_diff)

if pages_tree == "Загрузить набор аудио":
    st.markdown("### Аналитика набора аудио")
    if not ("login" in st.session_state and st.session_state.login is not None):
        st.write(f":red[Для того, чтобы загрузить набор аудио, войдите в аккаунт.]")
    else:
        st.markdown("Загрузите архив с файлами, для того, чтобы получить аналитику всех данных")
        uploaded_file = st.file_uploader("Архив с mp3/wav файлами", type=["zip"])
        if uploaded_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    st.success("Архив успешно загружен. Начинаем аналитику")

                    extracted_files = os.listdir(temp_dir)
                    st.write("Файлы в архиве:")
                    my_bar = st.progress(0, text="Анализ аудио моделью..")
                    tmp_analysis_res = []
                    for i, file in enumerate(extracted_files):
                        try:
                            format = "mp3" if "mp3" in file else "wav"
                            audio, sr = sf.read(os.path.join(temp_dir, file))
                            scores = get_audio_score(os.path.join(temp_dir, file))
                            tmp_analysis_res.append([[audio, sr], scores, file])

                            my_bar.progress(i / len(extracted_files), text="Анализ аудио моделью..")
                        except Exception as e:
                            continue
                    my_bar = st.empty()

            if len(tmp_analysis_res) > 0:
                st.success(f"Набор файлов загружен. Обнаружено и успешно обработано {len(tmp_analysis_res)} файлов.")
                st.write('Вы можете посмотреть аналитику по всем обработанным аудио в разделе "**Аналитика**"')
                st.session_state.global_data_pack_res = tmp_analysis_res.copy()
            else:
                st.write(f":red[Не удалось обработаь ни один файл.]")

if pages_tree == "Аналитика":
    if not ("login" in st.session_state and st.session_state.login is not None):
        st.write(f":red[Для того, чтобы посмотреть аналитику по набору аудио, войдите в аккаунт.]")
    elif "global_data_pack_res" not in st.session_state:
        st.write(f":red[Сперва загрузите набор аудио в предыдущем разделе.]")
    else:
        col1, col2 = st.columns((6, 4))
        with col1:
            data = {
                'Уровень картавости': [0, np.mean([x[1]["r_score"] for x in st.session_state.global_data_pack_res])],
                'Ошибки в буквах "Г"': [1, np.mean([x[1]["g_score"] for x in st.session_state.global_data_pack_res])],
                'Скорость чтения': [2, np.mean([x[1]["speed_score"] for x in st.session_state.global_data_pack_res])],
                'Общий рейтинг': [3, np.mean([x[1]["total_score"] for x in st.session_state.global_data_pack_res])]
            }
            df = pd.DataFrame(data).transpose()
            fig = px.bar(df, x=0, y=1, color=1, color_continuous_scale=generate_gradient(100),
                         title="Уровень ошибок за последние тренировки")
            fig.update_layout(title="Распределение ошибок в наборе загруженных аудио",
                              xaxis_title="Картавость     'ГЭканье'      Скорость чтения         Общий рейтинг",
                              yaxis_title="Уровень ошибки")
            st.plotly_chart(fig, use_container_width=True)

            best_audio = max(st.session_state.global_data_pack_res, key=lambda x: x[1]["total_score"])
            worst_audio = min(st.session_state.global_data_pack_res, key=lambda x: x[1]["total_score"])

        with col2:
            st.write(f':green[Аудио с самой лучшей метрикой:]')
            st.audio(best_audio[0][0], sample_rate=best_audio[0][1])
            with st.popover("Посмотреть детали"):
                scores_desription(best_audio[1], indents_count=40, key=0)

            st.write(f':red[Аудио с самой худшей метрикой:]')
            st.audio(worst_audio[0][0], sample_rate=worst_audio[0][1])
            with st.popover("Посмотреть детали"):
                scores_desription(worst_audio[1], indents_count=40, key=1)

        searcher = Searcher([x[2] for x in st.session_state.global_data_pack_res])
        selected_value = st_searchbox(
            searcher,
            key="Поиск по названию файла",
        )

        vals = st.session_state.global_data_pack_res.copy()
        if selected_value is not None:
            vals = [x for x in vals if x[2] == selected_value]

        st.session_state.selected_page = sac.pagination(align='center', jump=True, show_total=True,
                                                        total=len(vals), page_size=5)

        pg = st.session_state.selected_page
        user_audios_page = vals[(pg - 1) * 5:pg * 5]

        for k, audio in enumerate(user_audios_page):
            cont = st.container(border=True)
            scores = audio[1]
            col1, col2 = cont.columns((4, 6))
            col1.write(audio[2])
            cols_scores = col2.columns((3, 3, 4, 4))
            for i, (sc, sc_ru) in enumerate([("r_score", "Буква Р"), ("g_score", '"Гэканье"'),
                                  ("speed_score", "Скорость чтения"), ("total_score", "Общий рейтинг")]):
                if scores[sc] > 75:
                    color = "rgb(245, 108, 108)"
                elif scores[sc] > 45:
                    color = "rgb(240, 240, 100)"
                else:
                    color = "rgb(108, 245, 137)"
                with cols_scores[i]:
                    annotated_text((sc_ru, f"{scores[sc]}%", color))

            with cont.popover(label="Посмотреть детали", use_container_width=True):
                st.audio(audio[0][0], sample_rate=audio[0][1])
                scores_desription(scores, indents_count=54, key=str(k)+"_"+str(k))

if pages_tree == "Наши алгоритмы":
    st.markdown("## Наши алгоритмы")
    st.write("*...продолжают активно улучшаться, подробная информация в этом разделе появится очень скоро!*")

if pages_tree == "API":
    st.markdown("## API")
    st.write("""Мы реализовали открытый API интерфейс для получения предсказанного класса и логитов модели по mp3 файлу аудио.""")
    st.link_button("Ссылка на openapi", "https://phonix.pro/api/docs/")
    st.markdown("#### Протестировать API в сервисе")
    uploaded_file = st.file_uploader("Загрузите файл с аудио", type=["mp3"])
    if uploaded_file is not None:
        progress = st.empty()

        with progress:
            st.success("Происходит анализ аудио.. Пожалуйста, подождите немного")
        bytes_data = uploaded_file.getvalue()
        with open("tmp.mp3", "wb") as f:
            f.write(bytes_data)

        try:
            with open("tmp.mp3", "rb") as file:
                scores = requests.post(
                    "https://phonix.pro/api/predict",
                    files={"file": ("tmp.mp3", file, "audio/mpeg")}
                )
            st.write(json.loads(scores.content))
            progress = st.empty()
        except:
            with progress:
                st.warning("Произошла ошибка при анализе аудио..", )
    st.markdown("#### Code snippet для питона")
    st.code("""d = "my_audio"
fname = f"path/to/my/audio{d}.mp3"

 with open(fname, "rb") as file:
    response = requests.post(
        "https://phonix.pro/api/predict",
        files={"file": (f"{fname}.mp3", file, "audio/mpeg")},
        data={"device": device} 
    )

    result = json.loads(response.content)""", language="python", line_numbers=True)

if pages_tree == "Команда":
    st.markdown("#### Наша команда")
    st.write("""Команда: 
**Никита Ильтяков** (Москва, 18 лет)
- Первый курс ВШЭ ПМИ
- Победитель хакатона ЛЦТ 2024
- Победитель Национальной Олимпиады по Анализу Данных DANO
- Победитель Всероссийской Олимпиады Школьников по ИИ
- Трехкратный победитель хакатонов Цифровой Прорыв

**Александр Смирнов** (Санкт-Петербург, 17 лет)
- Первый курс ИТМО ИИИ
- Победитель Национальной Технологической Олимпиады по ИИ
- Победитель Национальной Олимпиады по Анализу Данных DANO
- Победитель AIIJC 2023

Мы любим решать интересные исследовательские задачи и находить неординарные решения проблем на пути. Мы являемся частью  RASCAR - активного сообщества талантливых молодых программистов, аналитиков и исследователей, в котором мы активно обмениваемся знаниями и опытом со всеми, кому интересен ИИ""")