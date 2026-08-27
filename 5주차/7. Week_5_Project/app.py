import asyncio

import streamlit as st

from agent.integrated_agent import run_integrated_agent


# ============================================================
# Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 통합 Agent",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# 제목
# ============================================================

st.title("AI 통합 Agent")

st.caption(
    "법률 RAG · 회원/판매 DB · Tavily 웹 검색"
)


# ============================================================
# 대화 기록 초기화
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 이전 대화 출력
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# ============================================================
# 사용자 입력
# ============================================================

question = st.chat_input(
    "질문을 입력하세요."
)


if question:

    # ----------------------------------------
    # 사용자 메시지 저장
    # ----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # ----------------------------------------
    # Agent 실행
    # ----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "답변을 생성하고 있습니다..."
        ):

            try:

                result = asyncio.run(
                    run_integrated_agent(
                        question
                    )
                )

                answer = result["result"]
                route = result["route"]

                st.markdown(answer)

                st.caption(
                    f"Route: {route}"
                )

            except Exception as e:

                answer = (
                    "Agent 실행 중 오류가 발생했습니다.\n\n"
                    f"`{e}`"
                )

                st.error(answer)


    # ----------------------------------------
    # Assistant 메시지 저장
    # ----------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    