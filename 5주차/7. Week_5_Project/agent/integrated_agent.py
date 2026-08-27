import os
import asyncio

from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from rag.legal_rag import (
    ask_legal_rag,
)

from database.db_agent import (
    create_mcp_agent,
    create_db_graph,
    run_db_agent,
)

from search.search_agent import (
    run_search_agent,
)

# ============================================================
# 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(
    BASE_DIR / ".env"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL"
)

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

class MainRouteDecision(BaseModel):

    route: Literal[
        "law",
        "database",
        "search"
    ] = Field(
        description=(
            "법률 질문이면 law, "
            "회원 또는 판매 DB 관련 질문이면 database, "
            "그 외 일반적인 정보 검색이면 search"
        )
    )

router_llm = llm.with_structured_output(
    MainRouteDecision
)

class MainState(TypedDict):

    query: str

    route: str

    result: str

async def router_node(
    state: MainState
):

    prompt = f"""
다음 사용자 요청을 세 종류 중 하나로 분류하세요.

[law]
법률이나 법적 규정에 관한 질문입니다.

현재 제공 가능한 법률:
- 개인정보 보호법
- 근로기준법
- 소득세법
- 주택임대차보호법

예:
- 근로시간은 몇 시간이야?
- 임차인의 대항력 조건은?
- 개인정보는 언제 파기해야 해?
- 소득세 납세지는 어디야?


[database]
회원 또는 판매 데이터에 대한 질문입니다.

예:
- 서울 회원 몇 명이야?
- 김철수의 구매 내역 알려줘.
- 새로운 회원 추가해줘.
- 1번 회원 연락처 수정해줘.
- 판매 기록 삭제해줘.


[search]
위 두 범주에 해당하지 않는 일반적인 정보나
인터넷 검색이 필요한 질문입니다.

예:
- 최근 NVIDIA 뉴스 알려줘.
- LangGraph 최신 기능 알려줘.
- 오늘 AI 산업 동향 알려줘.


사용자 요청:
{state["query"]}
"""

    decision = await router_llm.ainvoke(
        prompt
    )

    return {
        "route": decision.route
    }

async def law_node(
    state: MainState
):

    answer = await ask_legal_rag(
        state["query"]
    )

    return {
        "result": answer
    }

async def search_node(
    state: MainState
):

    answer = await run_search_agent(
        state["query"]
    )

    return {
        "result": answer
    }

def create_main_graph(
    db_graph
):

    async def database_node(
        state: MainState
    ):

        answer = await run_db_agent(
            db_graph,
            state["query"]
        )

        return {
            "result": answer
        }

    builder = StateGraph(
        MainState
    )

    # ----------------------------------------
    # Node
    # ----------------------------------------

    builder.add_node(
        "router",
        router_node
    )

    builder.add_node(
        "law",
        law_node
    )

    builder.add_node(
        "database",
        database_node
    )

    builder.add_node(
        "search",
        search_node
    )


    # ----------------------------------------
    # START
    # ----------------------------------------

    builder.add_edge(
        START,
        "router"
    )


    # ----------------------------------------
    # Router
    # ----------------------------------------

    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "law": "law",
            "database": "database",
            "search": "search",
        }
    )


    # ----------------------------------------
    # END
    # ----------------------------------------

    builder.add_edge(
        "law",
        END
    )

    builder.add_edge(
        "database",
        END
    )

    builder.add_edge(
        "search",
        END
    )


    return builder.compile()

async def ask_agent(
    graph,
    question: str
):

    result = await graph.ainvoke(
        {
            "query": question,
            "route": "",
            "result": "",
        }
    )

    print(
        "\n" + "=" * 70
    )

    print("[질문]")
    print(question)

    print("\n[최상위 Route]")
    print(result["route"])

    print("\n[답변]")
    print(result["result"])

    print(
        "=" * 70
    )

    return result

async def main():

    print(
        "통합 AI Agent 초기화 중..."
    )

    # ----------------------------------------
    # DB Agent 준비
    # ----------------------------------------

    mcp_agent = await create_mcp_agent()

    db_graph = create_db_graph(
        mcp_agent
    )


    # ----------------------------------------
    # 최상위 Graph 준비
    # ----------------------------------------

    main_graph = create_main_graph(
        db_graph
    )

    print(
        "통합 AI Agent 준비 완료"
    )


    # ----------------------------------------
    # 1. 법률 RAG 테스트
    # ----------------------------------------

    await ask_agent(
        main_graph,
        "근로자의 법정 근로시간은 몇 시간이야?"
    )


    # ----------------------------------------
    # 2. DB 테스트
    # ----------------------------------------

    await ask_agent(
        main_graph,
        "서울에 거주하는 회원은 몇 명이야?"
    )


    # ----------------------------------------
    # 3. Search 테스트
    # ----------------------------------------

    await ask_agent(
        main_graph,
        "최근 NVIDIA AI 관련 주요 뉴스를 알려줘."
    )

async def create_integrated_graph():

    # DB Agent 준비
    mcp_agent = await create_mcp_agent()

    db_graph = create_db_graph(
        mcp_agent
    )

    # Main Graph 준비
    main_graph = create_main_graph(
        db_graph
    )

    return main_graph
async def run_integrated_agent(
    question: str
) -> dict:

    graph = await create_integrated_graph()

    result = await graph.ainvoke(
        {
            "query": question,
            "route": "",
            "result": "",
        }
    )

    return result

if __name__ == "__main__":

    asyncio.run(
        main()
    )

