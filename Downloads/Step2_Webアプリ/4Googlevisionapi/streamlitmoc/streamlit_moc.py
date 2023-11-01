import streamlit as st
from google.cloud import vision
import os

# サービスアカウントキーファイルへのパスを設定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./my-project-playground-395801-b539ffd1fe38.json"

# Streamlitアプリのタイトル
st.title("画像チェックアプリ")

# ２行空行を追加
st.write("")
st.write("")

# 画像アップロード
uploaded_image = st.file_uploader("画像(jpg, png, jpeg)をアップロードしてください。", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # 画像を表示
    st.image(uploaded_image, caption="アップロードされた画像", use_column_width=True)

    # 判定ボタンがクリックされたら実行
    if st.button("判定"):
        # 画像ファイルをバイナリモードで読み込み
        content = uploaded_image.read()

        # Google Cloud Vision APIのクライアントを作成
        client = vision.ImageAnnotatorClient()

        # Vision APIに渡すために画像を設定
        image = vision.Image(content=content)

        # 画像のWeb検出を実行
        response = client.web_detection(image=image)

        # Web検出の結果を取得
        annotations = response.web_detection

        # 完全一致画像検出結果
        st.title("完全一致画像検出結果")
        if annotations.pages_with_matching_images:
            st.markdown('<font color="red">完全一致画像が検出されました。確認してください.</font>', unsafe_allow_html=True)
            for page in annotations.pages_with_matching_images:
                st.write(f"ウェブページURL: {page.url}")
        else:
            st.write("完全一致画像は検出されませんでした.")

        # ２行空行を追加
        st.write("")
        st.write("")

        # 類似画像検出結果
        st.title("類似画像検出結果")
        if annotations.visually_similar_images:
            st.markdown('<font color="yellow">類似画像が検出されました。確認してください.</font>', unsafe_allow_html=True)
            for image in annotations.visually_similar_images:
                st.write(f"画像URL: {image.url}")
        else:
            st.write("類似画像は検出されませんでした.")