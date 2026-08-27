"""소득세법 RAG 챗봇 Streamlit 화면."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from Chat_Bot import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LAW_FILE_PATH,
    ask_chatbot,
    get_agent_executor,
    get_vector_store,
)


st.set_page_config(page_title="소득세법 AI 챗봇", page_icon=":material/gavel:")


@st.cache_resource
def load_resources():
    """세션 재실행 때 LLM Agent와 Chroma 연결을 다시 만들지 않는다."""
    vector_store = get_vector_store()
    agent_executor = get_agent_executor()
    return vector_store, agent_executor


def to_langchain_history(messages: list[dict[str, str]]):
    history = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
    return history


st.session_state.setdefault("messages", [])
st.title("소득세법 AI 챗봇")
st.caption("2026년 7월 1일 시행 소득세법 문서 검색과 종합소득 산출세액 계산")

with st.sidebar:
    st.subheader("시스템 정보")
    st.write(f"문서: `{LAW_FILE_PATH.name}`")
    st.write(f"임베딩: `{EMBEDDING_MODEL}`")
    st.write(f"컬렉션: `{COLLECTION_NAME}`")
    if st.button("대화 지우기", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()
    st.info(
        "계산 결과는 종합소득 산출세액이며 세액공제·감면, 가산세, "
        "지방소득세는 별도입니다. 중요한 신고는 세무 전문가에게 확인하세요."
    )

try:
    with st.spinner("소득세법 검색 시스템을 준비하고 있습니다..."):
        load_resources()
except Exception as exc:
    st.error("챗봇 초기화에 실패했습니다.")
    st.exception(exc)
    st.stop()

if not st.session_state.messages:
    st.info(
        "예: ‘근로소득의 비과세 항목을 알려줘’ 또는 "
        "‘종합소득 과세표준 6천만원의 산출세액은?’"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("소득세법 또는 종합소득 산출세액을 질문하세요", submit_mode="disable")
if prompt:
    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 조문을 확인하고 있습니다..."):
                result = ask_chatbot(prompt, to_langchain_history(previous_messages))
            response = result.get("output", "답변을 생성하지 못했습니다.")
            st.markdown(response)
        except Exception as exc:
            response = f"요청을 처리하지 못했습니다. 설정과 서비스 상태를 확인해 주세요.\n\n`{exc}`"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
