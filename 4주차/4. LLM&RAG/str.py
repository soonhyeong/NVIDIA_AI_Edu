import streamlit as st

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.header("Tab 1")
    st.write("안녕하세요.")
    st.subheader("이곳은 Tab 1의 내용입니다.")

with tab2:
    st.header("Tab 2")
    st.write("This is the content of Tab 2.")

st.sidebar.title("사이드바 구성")
st.sidebar.write("여기는 사이드바 영역입니다.")
st.sidebar.checkbox("체크박스 예시")