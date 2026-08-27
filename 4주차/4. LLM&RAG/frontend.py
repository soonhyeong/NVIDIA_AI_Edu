# 스트리밋을 이용한 프론트 구현

import streamlit as st
import requests

# streamlit 기본 환경 설정
st.set_page_config(page_title='My AI Chabot', page_icon='🧠')
st.title('🧠 AI 챗봇')

# 대화기록 저장 리스트 생성
if 'messages' not in st.session_state:
    st.session_state.messages = []   

# 저장된 대화 기록을 하나씩 반복 출력
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt := st.chat_input("이전 대화 내용을 바탕으로 질문해보세요."):  # 사용자 입력란
    # 사용자 질문을 세션의 messages에 저장(새로고침 시 유지용)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 사용자 입력 메세지 화면에 즉시 랜더링
    with st.chat_message("user"):
        st.markdown(prompt)

    # 벡엔드 통신
    with st.chat_message("system"): # AI 답변 영역
        with st.spinner("맥락 파악 중..."):   # 답변이 나올때까지 화면에 로딩 아이콘 표시
            try:
                # 서버에 POST 요청
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"message": prompt},  # Json 형식으로 질문 전달
                    timeout=120  # 요청 제한 시간 설정
                )

                # 응답에 대한 처리
                if response.status_code == 200:
                    # 챗봇 응답결과 추출 화면 출력
                    answer = response.json().get("answer")
                    st.markdown(answer)
                    # 챗봇 응답결과를 세션에 저장
                    st.session_state.messages.append({"role": "system", "content": answer})
                else:
                    st.error("서버 응답 오류")
            except Exception as e:
                st.error(f"연결 실패: {e}")


# 실행: python -m streamlit run frontend.py