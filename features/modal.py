import streamlit as st
from streamlit_modal import Modal

st.set_page_config(layout="wide")

modal = Modal(title="まずご確認ください", key="demo_modal_key")

# ここに皆さんのmain.pyの中身が入るイメージ
st.title("テストページ")

# 初期起動時にモーダルを開く→サイドバーとの相性が悪いかも、、
if 'modal_opened' not in st.session_state:
    st.session_state.modal_opened = True
    modal.open()

# モーダルの状態をチェックして閉じる
if modal.is_open():
    with modal.container():
        # モーダル内に「×」印ボタンを追加し、クリックでモーダルを閉じる
        # close_button = st.button("閉じる")
        # if close_button:
        #     modal.close()


        # モーダル内にHTMLを直接書き込んで中央寄せにする
        st.write("お知らせ：社内画像利用ルールの一部改訂に関して　[こちら](https://tech0-jp.com/terms/)")
        st.write("最新の社内文書取り扱い[こちら](https://654fa2e0676e2a49fcd87dba--dreamy-sable-95c587.netlify.app/)")
        st.write("コンプライアンスセルフチェッカーのマニュアル　[こちら](https://tech0-jp.com/terms/)")
        st.write("-------------------------------------------------------------------------")
        st.title("**今週の要修正事項ランキング！！**\n")
        txt1 = '<p style="color:red;font-size: 30px;">1位：許可が下りていない社内画像を利用していた</p>'
        st.markdown(txt1, unsafe_allow_html=True)
        st.write("2位：景品表示法違反（優良誤認表示）")
        st.write("3位：他社商品の誹謗中傷")
