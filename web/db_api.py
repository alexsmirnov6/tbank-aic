import sqlite3


def add_user(login, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO users (login, password)
        VALUES (?, ?)
    ''', (login, password))

    conn.commit()
    conn.close()


def check_user(login, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT password FROM users WHERE login = ?', (login,))
    result = cursor.fetchone()

    conn.close()

    if result is None:
        return 0
    elif result[0] != password:
        return 1
    else:
        return 2


def login_exists(login):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT 1 FROM users WHERE login = ?', (login,))
    result = cursor.fetchone()

    conn.close()

    return result is not None


def add_user_audio(login, audio_hash, created_at):
    conn = sqlite3.connect('audios.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO users (login, audio_hash, created_at)
        VALUES (?, ?, ?)
    ''', (login, audio_hash, created_at))

    conn.commit()
    conn.close()


def get_audio_hashes_by_login(login):
    conn = sqlite3.connect('audios.db')
    cursor = conn.cursor()

    cursor.execute('''
            SELECT audio_hash, created_at FROM users WHERE login = ?
        ''', (login,))
    result = cursor.fetchall()

    conn.close()
    return [(row[0], row[1]) for row in result]



def add_score(audio_hash, r_score, g_score, stut_score, speed_score, total_score):
    conn = sqlite3.connect('scores.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO scores (audio_hash, r_score, g_score, stut_score, speed_score, total_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (audio_hash, r_score, g_score, stut_score, speed_score, total_score))

    conn.commit()
    conn.close()


def get_score_by_audio_hash(audio_hash):
    conn = sqlite3.connect('scores.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT audio_hash, r_score, g_score, stut_score, speed_score, total_score
        FROM scores
        WHERE audio_hash = ?
    ''', (audio_hash,))

    result = cursor.fetchone()
    conn.close()

    if result:
        print(result)
        return {
            'r_score': result[1],
            'g_score': result[2],
            "stut_score": result[3],
            'speed_score': result[4],
            'total_score': result[5]
        }
    else:
        return None


def clear_table(table_name):
    conn = sqlite3.connect(table_name)
    cursor = conn.cursor()

    try:
        cursor.execute(f'DELETE FROM users')
    except:
        cursor.execute(f"DELETE FROM scores")

    conn.commit()
    conn.close()


def delete_audio_by_hash_and_login(audio_hash, login):
    conn = sqlite3.connect('audios.db')
    cursor = conn.cursor()

    cursor.execute('''
        DELETE FROM users WHERE audio_hash = ? AND login = ?
    ''', (audio_hash, login))

    conn.commit()
    conn.close()

