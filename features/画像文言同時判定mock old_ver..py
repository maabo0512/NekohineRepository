import streamlit as st
import io
from PIL import Image
from PyPDF2 import PdfReader
from google.cloud import vision_v1
from google.oauth2 import service_account
import cv2
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Google Cloud Vision APIの認証情報を設定
# サービスアカウントキーのファイルパス（JSONファイル）を指定
credentials = service_account.Credentials.from_service_account_file("my-project-playground-395801-b539ffd1fe38.json")
client = vision_v1.ImageAnnotatorClient(credentials=credentials)

# NGワードとNG理由の辞書
ng_words_and_reasons = {
    "業界初": "調査・検証結果を示しましょう",
    "業界No.1": "調査・検証結果を示しましょう",
    "市場満足度●％": "調査・検証結果を示しましょう",
    "顧客満足度●％": "調査・検証結果を示しましょう",
    "一般的な": "現時点で一般的に流通しているか、改めて確認しましょう",
    "著しく": "効果を有する根拠（試験結果）も記載しましょう",
    "劇的に": "効果を有する根拠（試験結果）も記載しましょう",
    "インフルエンザウイルス": "具体的なウイルス名は記載できません",
    "●●菌": "具体的な菌名は記載できません",
    "具体的な社名": "製品比較の際に、具体的な社名の記載は避けてください"
}

# NG画像を設定
image_path_ng_image = 'train_original.jpg'
ng_image = cv2.imread(image_path_ng_image)
gray_ng_image = cv2.cvtColor(ng_image, cv2.COLOR_BGR2GRAY)

# イメージからテキストを抽出する関数
def extract_text_from_image(image):
    # 画像を読み込み
    image = Image.open(image)
    image = image.convert("RGB")
    content = io.BytesIO()
    image.save(content, format="JPEG")
    content = content.getvalue()

    # Google Cloud Vision APIを使用してテキスト抽出
    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    text = response.text_annotations[0].description
    return text

# 画像マッチングの関数
def image_matching(uploaded_image, ng_image):
    # 画像をグレースケールに変換
    gray_uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

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
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # 部分一致のしきい値を設定（任意の値）
    min_good_matches = 100

    return len(good_matches) >= min_good_matches

# アプリのタイトル
st.title("コンプラ・セルフチェッカー")

# 2行の空行を挿入
st.text("")  # 1つ目の空行
st.text("")  # 2つ目の空行

# ファイルアップロード
uploaded_document = st.file_uploader("画像またはPDFファイルをアップロードしてください", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_document:
    # 画像を表示
    if uploaded_document.type.startswith('image/'):
        st.image(uploaded_document, caption='アップロードされた画像', use_column_width=True)

    # 判定ボタン
    if st.button("判定"):
        text = ""
        is_ng_image_detected = False

        if uploaded_document.type == "application/pdf":
            # PDFからテキストを抽出
            pdf = PdfReader(uploaded_document)
            for page in pdf.pages:
                text += page.extract_text()
        else:
            # 画像からテキストを抽出
            text = extract_text_from_image(uploaded_document)

        # テキストを文に分割
        sentences = sent_tokenize(text)

        # 画像またはPDFから読み取ったテキストを表示
        st.subheader("テキスト抽出結果:")
        st.write(text)

        # NGワードが検出されたかどうかを示すフラグ
        ng_word_detected = False

        # 各NGカテゴリーのNGワードと理由を格納するリスト
        ng_categories = {}

        # 各文に対してNGワードを検出
        for sentence in sentences:
            for ng_word, ng_reason in ng_words_and_reasons.items():
                if ng_word in sentence:
                    ng_word_detected = True
                    # NGカテゴリーにNGワードと理由を追加
                    if ng_word not in ng_categories:
                        ng_categories[ng_word] = [ng_reason]
                    else:
                        ng_categories[ng_word].append(ng_reason)

        # 2行の空行を挿入
        st.text("")  # 1つ目の空行
        st.text("")  # 2つ目の空行

        # NGワードが検出された場合、各NGカテゴリーごとに表示
        # 判定結果のセクションを追加
        st.subheader("テキスト判定結果")
        if ng_word_detected:
            for ng_word, ng_reasons in ng_categories.items():
                st.write(f"NGワード: {ng_word}")
                st.write("NG理由:", ', '.join(ng_reasons))
        else:
            st.write("NGワードは検出されませんでした.")

        # 2行の空行を挿入
        st.text("")  # 1つ目の空行
        st.text("")  # 2つ目の空行

        # 画像結果を表示
        # 判定結果のセクションを追加
        st.subheader("画像確認結果")
        if uploaded_document.type.startswith('image/'):
            # NG画像の検出
            is_ng_image_detected = image_matching(ng_image, ng_image)
            if is_ng_image_detected:
                st.write("使用禁止の画像が下記の通り検出されています。確認してください。")
                st.image(ng_image, caption='NG画像が検出されました', use_column_width=True)