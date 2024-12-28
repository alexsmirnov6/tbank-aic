import streamlit as st
from db_api import add_user, check_user, login_exists


@st.dialog("Регистрация")
def signup():
    login = st.text_input("Придумайте имя пользоватея")
    password = st.text_input("Придумайте пароль", type="password")
    password2 = st.text_input("Подтвердите пароль", type="password")

    if st.button("Зарегистрироваться"):
        if login_exists(login):
            st.write(":red[Пользователь с таким логином уже существует]")
        else:
            if password2 == password and len(password) > 5:
                st.session_state["start_register"] = False
                add_user(login, password)
                st.rerun()
            elif password2 != password:
                st.write(":red[Пароль не совпадает]")
            elif len(password) <= 5:
                st.write(":red[Длина пароля должна быть больше пяти символов]")


@st.dialog("Вход")
def signin():
    login = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type="password")
    col1, col2 = st.columns((2, 9))
    if col1.button("Войти", key="login_1"):
        signin_status = check_user(login, password)
        if signin_status == 0:
            st.write(":red[Пользователь не найден]")
        elif signin_status == 1:
            st.write(":red[Пароль неверный]")
        else:
            st.session_state["login"] = login
            st.rerun()
    if col2.button("Регистрация", key="register_1"):
        st.session_state["start_register"] = True
        st.rerun()
