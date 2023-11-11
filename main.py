import streamlit as st
import sqlite3
import pandas as pd
import datetime
from PIL import Image
from PyPDF2 import PdfReader
from google.cloud import vision_v1
from google.oauth2 import service_account
import io
import nltk
from nltk.tokenize import sent_tokenize

# モックのユーザー情報
MOCK_USER_INFO = {
    "username": "nekohineri",
    "password": "neko",
    "role": "管理者"
}

# データベースの設定とテーブル作成
def setup_database():
    conn = sqlite3.connect('app_data.db', check_same_thread=False)
    c = conn.cursor()
    # 既存のテーブルの作成
    c.execute('''CREATE TABLE IF NOT EXISTS users (ID INTEGER PRIMARY KEY, 名前 TEXT, 役割 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ng_words (ID INTEGER PRIMARY KEY, NGワード TEXT, 警告文 TEXT, 関連法令規定 TEXT, 登録日時 TEXT, 登録者 TEXT)''')
    conn.commit()
    return conn, c

# ログイン機能
def login(username, password):
    return username == MOCK_USER_INFO['username'] and password == MOCK_USER_INFO['password']

# メイン関数
def main():
    conn, c = setup_database()

    st.title("コンプラ・セルフチェッカー")

    if 'logged_in' not in st.session_state:
        display_login()

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        display_navigation(conn, c)

# ログイン画面の表示
def display_login():
    with st.container():
        username = st.text_input("ユーザー名")
        password = st.text_input("パスワード", type="password")
        if st.button("ログイン"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['role'] = MOCK_USER_INFO['role']
                st.session_state['main_page'] = "判定画面"
                st.success("ログインに成功しました！")
            else:
                st.error("ログインに失敗しました。ユーザー名、パスワードを確認してください。")

# ナビゲーションの表示
def display_navigation(conn, c):
    if st.session_state['role'] == "管理者":
        st.session_state['main_page'] = st.sidebar.radio("メインページ", ["判定画面", "管理画面"])
    else:
        st.session_state['main_page'] = "判定画面"

    if st.session_state['main_page'] == "管理画面":
        selected_option = st.sidebar.radio("管理メニュー", ["ユーザー管理", "NGワード管理", "NG画像管理", "レポート・統計", "監査ログ", "設定"])
        st.header(selected_option)
        manage_content(selected_option, conn, c)
    elif st.session_state['main_page'] == "判定画面":
        display_check_screen(conn, c)

# 判定画面の表示
def display_check_screen(conn, c):
    st.subheader("判定画面")
    # 判定画面の詳細な処理をここに記述
    # Google Cloud Vision APIの認証情報を設定
    credentials = service_account.Credentials.from_service_account_file("credentials.json")
    client = vision_v1.ImageAnnotatorClient(credentials=credentials)

    # データベースからNGワードを取得する関数
    def get_ng_words_from_db():
        c.execute("SELECT NGワード, 警告文, 関連法令規定 FROM ng_words")
        ng_words_data = c.fetchall()
        return {ng_word: (warning, law) for ng_word, warning, law in ng_words_data}

    # NGワードをデータベースから取得
    ng_words_and_warnings = get_ng_words_from_db()

    # イメージからテキストを抽出する関数
    def extract_text_from_image(image):
        # 画像を読み込み
        image = Image.open(image)

        # 画像をRGBモードに変換
        image = image.convert("RGB")
        content = io.BytesIO()
        image.save(content, format="JPEG")
        content = content.getvalue()

        # Google Cloud Vision APIを使用してテキスト抽出
        image = vision_v1.Image(content=content)
        response = client.text_detection(image=image)
        text = response.text_annotations[0].description
        return text

    # ファイルアップロード
    uploaded_file = st.file_uploader("画像またはPDFファイルをアップロードしてください", type=["jpg", "png", "jpeg", "pdf"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = ""
            # PDFからテキストを抽出
            pdf = PdfReader(uploaded_file)
            for page in pdf.pages:
                text += page.extract_text()
        else:
            # 画像からテキストを抽出
            text = extract_text_from_image(uploaded_file)

            # 画像を表示
            st.image(uploaded_file, caption='アップロードされた画像', use_column_width=True)

        # 判定ボタン
        if st.button("判定"):
            # テキストを文に分割
            sentences = sent_tokenize(text)

            # 画像またはPDFから読み取ったテキストを表示
            st.subheader("画像またはPDFから読み取ったテキスト:")
            st.write(text)

            # NGワードが検出されたかどうかを示すフラグ
            ng_word_detected = False

            # 各NGカテゴリーのNGワードと理由を格納するリスト
            ng_categories = {}

            # 各文に対してNGワードを検出
            for sentence in sentences:
                for ng_word, (warning, law) in ng_words_and_warnings.items():
                    if ng_word in sentence:
                        ng_word_detected = True
                        if ng_word not in ng_categories:
                            ng_categories[ng_word] = (warning, law)
                        else:
                            ng_categories[ng_word] = ng_categories[ng_word] + (warning, law)

            # 判定結果のセクションを追加
            st.subheader("判定結果")

            # NGワードが検出された場合、各NGカテゴリーごとに表示
            if ng_word_detected:
                for ng_word, (warnings, laws) in ng_categories.items():
                    st.write(f"NGワード: {ng_word}")
                    st.write("警告文:", warnings)
                    st.write("関連法令規定:", laws)
            else:
                st.write("NGワードは検出されませんでした.")
    else:
        st.write("ファイルがアップロードされていません。")

# 管理コンテンツの表示
def manage_content(selected_option, conn, c):
    if selected_option == "ユーザー管理":
        manage_users(conn, c)
    elif selected_option == "NGワード管理":
        manage_ng_words(conn, c)
    # 以下、他の管理機能に関する処理

# ユーザー管理の処理
def manage_users(conn, c):
    display_user_list(conn, c)
    if add_new_user(conn, c) or delete_user(conn, c):
        display_user_list(conn, c)

def display_user_list(conn, c):
    c.execute("SELECT * FROM users")
    user_data = c.fetchall()
    user_df = pd.DataFrame(user_data, columns=['ID', '名前', '役割']).set_index('ID')
    st.subheader("ユーザー一覧")
    st.table(user_df)

def add_new_user(conn, c):
    st.subheader("新しいユーザーを追加")
    new_user_name = st.text_input("名前を入力してください:")
    new_user_role = st.selectbox("役割を選択してください:", ["管理者", "編集者", "閲覧者"])
    add_button = st.button("追加")
    if add_button and new_user_name:
        c.execute("INSERT INTO users (名前, 役割) VALUES (?, ?)", (new_user_name, new_user_role))
        conn.commit()
        st.success(f"{new_user_name}さんを追加しました。")
        return True
    return False

def delete_user(conn, c):
    st.subheader("ユーザーを削除")
    c.execute("SELECT 名前 FROM users")
    users = c.fetchall()
    delete_user_name = st.selectbox("削除するユーザーを選択してください:", [user[0] for user in users])
    delete_user_button = st.button("削除")
    if delete_user_button and delete_user_name:
        c.execute("DELETE FROM users WHERE 名前=?", (delete_user_name,))
        conn.commit()
        st.success(f"{delete_user_name}さんを削除しました。")
        return True
    return False

# NGワード管理の処理
def manage_ng_words(conn, c):
    display_ng_words_list(conn, c)
    if add_new_ng_word(conn, c) or delete_ng_word(conn, c):
        display_ng_words_list(conn, c)

def display_ng_words_list(conn, c):
    # テーブルの列構造を確認
    c.execute("PRAGMA table_info(ng_words);")
    columns_info = c.fetchall()
    # 列名のリストを作成
    column_names = [column[1] for column in columns_info]

    # NGワードのデータを取得
    c.execute("SELECT * FROM ng_words")
    ng_word_data = c.fetchall()

    # 列名に基づいてDataFrameを作成
    ng_word_df = pd.DataFrame(ng_word_data, columns=column_names).set_index('ID')
    st.subheader("NGワード一覧")
    st.table(ng_word_df)

def add_new_ng_word(conn, c):
    st.subheader("新しいNGワードを追加")
    new_ng_word = st.text_input("NGワードを入力してください:")
    new_warning_text = st.text_input("警告文を入力してください:")
    new_related_laws = st.text_input("関連法令・規定を入力してください:")
    add_ng_button = st.button("追加")
    if add_ng_button and new_ng_word:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO ng_words (NGワード, 警告文, 関連法令規定, 登録日時, 登録者) VALUES (?, ?, ?, ?, ?)",
        (new_ng_word, new_warning_text, new_related_laws, now, "管理者"))
        conn.commit()
        st.success(f"NGワード「{new_ng_word}」を追加しました。")
        return True
    return False

def delete_ng_word(conn, c):
    st.subheader("NGワードを削除")
    c.execute("SELECT NGワード FROM ng_words")
    ng_words = c.fetchall()
    delete_ng_word = st.selectbox("削除するNGワードを選択してください:", [word[0] for word in ng_words])
    delete_ng_button = st.button("削除")
    if delete_ng_button and delete_ng_word:
        c.execute("DELETE FROM ng_words WHERE NGワード=?", (delete_ng_word,))
        conn.commit()
        st.success(f"NGワード「{delete_ng_word}」を削除しました。")
        return True
    return False

# 以下、他の管理機能に関する詳細な処理を追加

if __name__ == "__main__":
    main()