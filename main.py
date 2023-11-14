import streamlit as st
from streamlit_modal import Modal
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
import cv2
import numpy as np

# 画像マッチング関数
def image_matching(uploaded_image, ng_image_path='train_original.jpg'):
    # NG画像を読み込む
    ng_image = cv2.imread(ng_image_path)
    gray_ng_image = cv2.cvtColor(ng_image, cv2.COLOR_BGR2GRAY)

    # アップロードされた画像をOpenCV形式に変換
    uploaded_image_np = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # 画像をグレースケールに変換
    gray_uploaded_image = cv2.cvtColor(uploaded_image_np, cv2.COLOR_BGR2GRAY)

    # SIFT特徴点検出器を初期化
    sift = cv2.SIFT_create()

    # 特徴点と特徴記述子を抽出
    keypoints_ng_image, descriptors_ng_image = sift.detectAndCompute(gray_ng_image, None)
    keypoints_uploaded_image, descriptors_uploaded_image = sift.detectAndCompute(gray_uploaded_image, None)

    # マッチングアルゴリズムを選択
    bf = cv2.BFMatcher()

    # 特徴記述子をマッチング
    matches = bf.knnMatch(descriptors_ng_image, descriptors_uploaded_image, k=2)

    # マッチングの閾値を設定
    ratio = 0.75
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

    # 部分一致のしきい値を設定（任意の値）
    min_good_matches = 100

    # NG画像の検出結果と画像のパスを返す
    return len(good_matches) >= min_good_matches, ng_image_path

# モーダルの初期化
my_modal = Modal(title="まずはこちらをご確認ください", key="demo_modal_key", max_width=720)

# 初期起動時にモーダルを開く
if 'modal_opened' not in st.session_state:
    st.session_state.modal_opened = True
    my_modal.open()

# モーダルの状態をチェックして表示
if my_modal.is_open():
    with my_modal.container():
        # モーダル内のコンテンツ
        st.write("お知らせ：社内画像利用ルールの一部改訂に関して　[こちら](https://tech0-jp.com/terms/)")
        st.write("最新の社内文書取り扱い[こちら](https://dreamy-sable-95c587.netlify.app/)")
        st.write("コンプライアンスセルフチェッカーのマニュアル　[こちら](https://tech0-jp.com/terms/)")
        st.write("-------------------------------------------------------------------------")
        st.title("**今週の要修正事項ランキング！**\n")
        txt1 = '<p style="color:red;font-size: 30px;"><b>1位：許可が下りていない社内画像を利用していた</b></p>'
        st.markdown(txt1, unsafe_allow_html=True)
        st.write("2位：景品表示法違反（優良誤認表示）")
        st.write("3位：他社商品の誹謗中傷")

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

    # テーブルの存在確認と作成
    c.execute('''CREATE TABLE IF NOT EXISTS users (ID INTEGER PRIMARY KEY, 名前 TEXT, 役割 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ng_words (
        ID INTEGER PRIMARY KEY, 
        NGワード TEXT, 
        登録日時 TEXT, 
        登録者 TEXT
    )''')

    # ng_words テーブルの列情報を取得
    c.execute("PRAGMA table_info(ng_words);")
    columns_info = c.fetchall()
    columns_names = [column[1] for column in columns_info]

    # 警告文 列がない場合、列を追加
    if '警告文' not in columns_names:
        c.execute("ALTER TABLE ng_words ADD COLUMN 警告文 TEXT")

    # 関連法令規定 列がない場合、列を追加
    if '関連法令規定' not in columns_names:
        c.execute("ALTER TABLE ng_words ADD COLUMN 関連法令規定 TEXT")

    conn.commit()
    return conn, c

# ログイン機能
def login(username, password):
    return username == MOCK_USER_INFO['username'] and password == MOCK_USER_INFO['password']

# メイン関数
def main():
    conn, c = setup_database()

    st.title("コンプラ・セルフチェッカー")

    # ログイン状態に基づいて、ログイン画面を表示するかどうかを決定
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
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
                # ログイン成功後に画面を更新してログインフォームを非表示にする
                st.experimental_rerun()
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

            # NGワード判定結果のセクションを追加
            st.subheader("NGワード判定結果")

            # NGワードが検出された場合、各NGカテゴリーごとに表示
            if ng_word_detected:
                for ng_word, (warnings, laws) in ng_categories.items():
                    st.write(f"NGワード: {ng_word}")
                    st.write("警告文:", warnings)
                    st.write("関連法令規定:", laws)
            else:
                st.write("NGワードは検出されませんでした.")

            # NGワード判定結果のセクションを追加
            st.subheader("NG画像判定結果")

            # 画像マッチングを実行
            if uploaded_file.type.startswith('image/'):
                is_ng_image_detected, detected_ng_image_path = image_matching(uploaded_file)
                if is_ng_image_detected:
                    st.write("使用禁止の画像が下記の通り検出されています。確認してください。")
                    st.image(detected_ng_image_path, caption='検出されたNG画像', use_column_width=True)
                else:
                    st.write("画像に問題はありません。")
    else:
        st.write("ファイルがアップロードされていません。")

# 管理コンテンツの表示
def manage_content(selected_option, conn, c):
    if selected_option == "ユーザー管理":
        manage_users(conn, c)
    elif selected_option == "NGワード管理":
        manage_ng_words(conn, c)
    elif selected_option == "NG画像管理":
        manage_ng_images(conn, c)
    # 以下、他の管理機能に関する処理

    # ユーザー管理の処理
def manage_users(conn, c):
    if add_new_user(conn, c) or delete_user(conn, c):
        pass
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
    if add_new_ng_word(conn, c) or delete_ng_word(conn, c):
        pass
    display_ng_words_list(conn, c)

def add_new_ng_word(conn, c):
    st.subheader("新しいNGワードを追加")
    new_ng_word = st.text_input("NGワードを入力してください:")
    new_warning_text = st.text_input("警告文（NG理由）を入力してください:")
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

# NG画像管理の処理
def manage_ng_images(conn, c):
    if add_new_ng_image(conn, c) or delete_ng_image(conn, c):
        pass
    display_ng_images_list(conn, c)

def create_thumbnail(image, max_size=(300, 300)):
    """
    アップロードされた画像からサムネイルを作成し、指定された最大サイズに合わせて縮小する
    """
    img = Image.open(image)
    # RGBAモードの画像をRGBモードに変換
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.thumbnail(max_size, Image.Resampling.LANCZOS)  # ANTIALIASからResampling.LANCZOSに変更
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img.save(img_byte_arr, format='JPEG', quality=85)  # JPE    Gの品質を設定
    return img_byte_arr.getvalue()

def add_new_ng_image(conn, c):
    st.subheader("新しいNG画像を追加")
    uploaded_image = st.file_uploader("NG画像をアップロードしてください:", type=["jpg", "png", "jpeg"])
    new_ng_image_title = st.text_input("NG画像タイトルを入力してください:")
    new_warning_text = st.text_input("警告文（NG理由）を入力してください:")
    add_ng_image_button = st.button("追加")
    if add_ng_image_button and uploaded_image:
        # サムネイルを作成
        thumbnail = create_thumbnail(uploaded_image)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO ng_images (NG画像, NG画像タイトル, 警告文, 登録日時, 登録者) VALUES (?, ?, ?, ?, ?)",
                (thumbnail, new_ng_image_title, new_warning_text, now, "管理者"))
        conn.commit()
        st.success(f"NG画像「{new_ng_image_title}」を追加しました。")
        return True
    return False

def delete_ng_image(conn, c):
    st.subheader("NG画像を削除")
    c.execute("SELECT ID, NG画像タイトル FROM ng_images")
    ng_images = c.fetchall()

    # 画像が存在しない場合、処理を中断
    if not ng_images:
        st.write("削除するNG画像がありません。")
        return False

    delete_ng_image_option = st.selectbox(
        "削除するNG画像を選択してください:",
        ng_images,
        format_func=lambda x: f"{x[0]} - {x[1]}"  # IDとタイトルの組み合わせ
    )

    if delete_ng_image_option is not None:
        delete_ng_image_id = delete_ng_image_option[0]  # タプルからIDを取得
        delete_ng_image_button = st.button("削除")
        if delete_ng_image_button:
            c.execute("DELETE FROM ng_images WHERE ID=?", (delete_ng_image_id,))
            conn.commit()
            st.success(f"NG画像ID: {delete_ng_image_id} が削除されました。")
            return True
    return False

def display_ng_images_list(conn, c):
    c.execute("SELECT * FROM ng_images")
    ng_image_data = c.fetchall()
    
    # 画像データは表示できないため、列名を適切に調整
    ng_image_df = pd.DataFrame(ng_image_data, columns=['ID', 'NG画像', 'NG画像タイトル', '警告文', '登録日時', '登録者']).set_index('ID')
    st.subheader("NG画像一覧")
    st.table(ng_image_df.drop(columns=['NG画像']))  # 画像データは除外して表示

    # 一行に表示するサムネイルの数を設定
    thumbnails_per_row = 4

    # 画像データの変換と表示
    for i in range(0, len(ng_image_data), thumbnails_per_row):
        cols = st.columns(thumbnails_per_row)
        for j in range(thumbnails_per_row):
            if i + j < len(ng_image_data):
                image_data = ng_image_data[i + j]
                image = Image.open(io.BytesIO(image_data[1]))
                # `use_column_width`をTrueに設定して画像が列幅に合わせて調整されるようにする
                cols[j].image(image, caption=image_data[2], use_column_width=True)

# データベースの設定とテーブル作成の部分にNG画像テーブルの作成を追加
def setup_database():
    conn = sqlite3.connect('app_data.db', check_same_thread=False)
    c = conn.cursor()
    # ... 既存のテーブル作成コード ...
    c.execute('''CREATE TABLE IF NOT EXISTS ng_images (
        ID INTEGER PRIMARY KEY,
        NG画像 BLOB,
        NG画像タイトル TEXT,
        警告文 TEXT,
        登録日時 TEXT,
        登録者 TEXT
    )''')
    conn.commit()
    return conn, c

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


# 以下、他の管理機能に関する詳細な処理を追加

if __name__ == "__main__":
    main()
