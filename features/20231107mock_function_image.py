import streamlit as st
import cv2
import numpy as np

# NG画像を設定
image_path_ng_image = 'train_original.jpg'
ng_image = cv2.imread(image_path_ng_image)
gray_ng_image = cv2.cvtColor(ng_image, cv2.COLOR_BGR2GRAY)

# NG画像をBGRからRGBに変換
ng_image_rgb = cv2.cvtColor(ng_image, cv2.COLOR_BGR2RGB)

# 画像マッチングの関数
def image_matching(uploaded_image, ng_image):

    # アップロードされた画像をOpenCV形式に変換
    uploaded_image_np = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

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
uploaded_image = st.file_uploader("画像またはPDFファイルをアップロードしてください", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_image:
    # 画像を表示
    if uploaded_image.type.startswith('image/'):
        st.image(uploaded_image, caption='アップロードされた画像', use_column_width=True)

    # 判定ボタン
    if st.button("判定"):
        text = ""
        is_ng_image_detected = False

        # 画像結果を表示
        # 判定結果のセクションを追加
        st.subheader("画像確認結果")
        if uploaded_image.type.startswith('image/'):
            # NG画像の検出
            is_ng_image_detected = image_matching(uploaded_image, ng_image)
            if is_ng_image_detected:
                st.write("使用禁止の画像が下記の通り検出されています。確認してください。")
                st.image(ng_image_rgb, caption='NG画像が検出されました', use_column_width=True)
            else:
                st.write("問題無しです！")