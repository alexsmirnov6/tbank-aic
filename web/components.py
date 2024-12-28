import streamlit_antd_components as sac
import streamlit as st
import plotly.express as px
from texts import *
from fuzzywuzzy import fuzz


container_style = """
    <style>
        .container1 {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .container2 {
            /* Add styles for Container 2 if needed */
        }
    </style>
"""


class Searcher:
    def __init__(self, search_list):
        self.search_list = search_list

    def __call__(self, searchitem):
        res = []
        for s in self.search_list:
            if fuzz.partial_ratio(s, searchitem) > 75:
                res.append(s)
        return res




def scores_desription(scores, indents_count=60, key=None):
    st.write(":heavy_minus_sign:" * indents_count)
    for sc in ["r_score", "g_score", "stut_score", "speed_score", "total_score"]:

        if scores[sc] > 75:
            color = "red"
        elif scores[sc] > 45:
            color = "orange"
        else:
            color = "green"

        data = dict(
            categories=[str(scores[sc]) + "%", '  ', ' '],
            parent=['', str(scores[sc]) + "%", str(scores[sc]) + "%"],
            value=[100, scores[sc], 100 - scores[sc]])
        fig = px.sunburst(data,
                          names='categories',
                          parents='parent',
                          values='value',
                          branchvalues='total',
                          color_discrete_sequence=[color],
                          width=240)
        fig.update_layout(font=dict(size=50),
                          margin=dict(t=0, b=0))
        fig.update_traces(insidetextfont=dict(size=1))

        col1, col2 = st.columns((4, 7))
        col1.plotly_chart(fig, use_container_width=False, key=f"{str(key)}_{sc}")
        st.write(":heavy_minus_sign:" * indents_count)
        if sc == "r_score":
            col2.markdown("### Степень картавости")
            col2.write(f":{color}[{R_RATING[color]}]")
        elif sc == "g_score":
            col2.markdown("### Ошибки в произношении буквы 'Г'")
            col2.write(f":{color}[{G_RATING[color]}]")
        elif sc == "speed_score":
            col2.markdown("### Темп речи")
            col2.write(f":{color}[{SPEED_RATING[color]}]")
        elif sc == "stut_score":
            col2.markdown("### Заикание")
            col2.write(f":{color}[{STUT_RATING[color]}]")
        else:
            col2.markdown("### Итоговый рейтинг")
            col2.write(f":{color}[{TOTAL_RATING[color]}]")
        col2.write(EXCEPTION_TYPE_SCORE[sc])


def generate_gradient(n, lightness_factor=0.5):
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)

    gradient = []

    def lighten_color(color, factor):
        return tuple(int(c + (255 - c) * factor) for c in color)

    for i in range(n):
        t = i / (n - 1)

        if t <= 0.5:
            t_green_yellow = t / 0.5
            r = int(green[0] + t_green_yellow * (yellow[0] - green[0]))
            g = int(green[1] + t_green_yellow * (yellow[1] - green[1]))
            b = int(green[2] + t_green_yellow * (yellow[2] - green[2]))
        else:
            t_yellow_red = (t - 0.5) / 0.5
            r = int(yellow[0] + t_yellow_red * (red[0] - yellow[0]))
            g = int(yellow[1] + t_yellow_red * (red[1] - yellow[1]))
            b = int(yellow[2] + t_yellow_red * (red[2] - yellow[2]))

        light_color = lighten_color((r, g, b), lightness_factor)
        gradient.append([t, f"rgb({light_color[0]},{light_color[1]},{light_color[2]})"])

    return gradient


def show_progress_text(data, r_diff, g_diff, stut_score, speed_diff, total_diff):
    if abs(r_diff) < 5:
        if r_diff < 0:
            st.write(
                f'Ваш прогресс в произношении буквы "Р" за последнее время ухудшился на :red[{abs(r_diff)}%]. Рекомендуем включить в тренировки проверку картавости.')
        else:
            st.write(
                f'Ваш прогресс в произношении буквы "Р" за последнеее время улучшился на :green[{r_diff}]%. Продолжайте в том же духе!')
    else:
        if data["Уровень картавости"][1] > 75:
            st.write('Ваш прогресс в произношении буквы "Р" стабильно высокий. Продолжайте в том же духе!')
        else:
            st.write(
                'Ваш прогресс в произношении буквы "Р" не изменился за последнее время. **Рекомендуем повысить уровень сложности.**')

    if abs(g_diff) < 5:
        if g_diff < 0:
            st.write(
                f'Ваш прогресс в произношении буквы "Г" за последнее время ухудшился на :red[{abs(g_diff)}%]. Рекомендуем включить в тренировки проверку произношения буквы "Г".')
        else:
            st.write(
                f'Ваш прогресс в произношении буквы "Г" за последнеее время улучшился на :green[{g_diff}]%. Продолжайте в том же духе!')
    else:
        if data['Ошибки в буквах "Г"'][1] > 75:
            st.write('Ваш прогресс в произношении буквы "Г" стабильно высокий. Продолжайте в том же духе!')
        else:
            st.write(
                'Ваш прогресс в произношении буквы "Г" не изменился за последнее время. **Рекомендуем повысить уровень сложности.**')

    if abs(stut_score) < 5:
        if stut_score < 0:
            st.write(
                f'Ваш прогресс в борьбе с заиканием за последнее время ухудшился на :red[{abs(g_diff)}%]. Рекомендуем включить в тренировки проверку Заикания')
        else:
            st.write(
                f'Ваш прогресс в борьбе с заиканием за последнеее время улучшился на :green[{g_diff}]%. Продолжайте в том же духе!')
    else:
        if stut_score['Заикание'][1] > 75:
            st.write('Ваш прогресс в отсутствии заикания стабильно высокий. Продолжайте в том же духе!')
        else:
            st.write(
                'Ваш прогресс в борьбе с заиканием не изменился за последнее время. **Рекомендуем повысить уровень сложности.**')

    if abs(speed_diff) < 5:
        if speed_diff < 0:
            st.write(
                f'Ваш прогресс в быстрочтении за последнее время ухудшился на :red[{abs(speed_diff)}%]. Рекомендуем тренировать скорость чтения, чтобы повышать качество оценки речи нашими алгоритмами.')
        else:
            st.write(
                f'Ваш прогресс в быстрочтении за последнеее время улучшился на :green[{speed_diff}]%. Продолжайте в том же духе!')
    else:
        if data["Скорость чтения"][1] > 75:
            st.write('Ваш прогресс в быстрочтении стабильно высокий. Продолжайте в том же духе!')
        else:
            st.write(
                'Ваш прогресс в быстрочтении не изменился за последнее время. **Рекомендуем повысить уровень сложности.**')

    if abs(total_diff) < 5:
        if total_diff < 0:
            st.write(
                f'Ваш общий прогресс за последнее время ухудшился на :red[{abs(total_diff)}%]. Рекомендуем больше тренироваться или, при необходимости, понизить уровень сложности.')
        else:
            st.write(
                f'Ваш общий прогресс за последнеее время улучшился на :green[{total_diff}]%. Продолжайте в том же духе!')
    else:
        if data["Общий рейтинг"][1] > 75:
            st.write('Ваш суммарный прогресс стабильно высокий. Продолжайте в том же духе!')
        else:
            st.write(
                'Ваш суммарный прогресс не изменился за последнее время. **Рекомендуем повысить уровень сложности.**')