import os
import asyncio

from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END

# ============================================================
# 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

PYTHON_PATH = r"C:\miniconda\envs\langchain_up\python.exe"

MCP_SERVER_PATH = str(
    BASE_DIR / "database" / "mcp_server.py"
)

DATABASE_URL = (
    f"mysql+mysqlconnector://"
    f"{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}/{MYSQL_DATABASE}"
)

db = SQLDatabase.from_uri(
    DATABASE_URL,
    include_tables=[
        "user_project",
        "sale_project"
    ],
    sample_rows_in_table_info=3,
)

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

SQL_SYSTEM_PROMPT = """
당신은 MySQL 데이터베이스 조회 전용 Agent입니다.

사용 가능한 테이블:
- user_project
- sale_project

규칙:

1. 데이터 조회, 검색, 통계, 집계만 수행하세요.
2. 반드시 SELECT만 사용하세요.
3. INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE는 절대 실행하지 마세요.
4. user_project와 sale_project는 user_id로 JOIN할 수 있습니다.
5. 필요한 컬럼만 조회하세요.
6. pw 컬럼은 조회하거나 출력하지 마세요.
7. SQL 실행 전 쿼리를 확인하세요.
8. DB에 존재하지 않는 정보는 추측하지 마세요.
9. 최종 결과는 이해하기 쉬운 한국어로 설명하세요.
"""

sql_toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

sql_tools = sql_toolkit.get_tools()

sql_agent = create_agent(
    model=llm,
    tools=sql_tools,
    system_prompt=SQL_SYSTEM_PROMPT,
)

MCP_SYSTEM_PROMPT = """
당신은 회원 및 판매 정보를 관리하는 데이터베이스 관리 Agent입니다.

MCP Tool을 이용하여 데이터 등록, 수정, 삭제 작업을 수행하세요.

[회원]
- create_user
- get_user
- update_user
- delete_user

[판매]
- create_sale
- get_sale
- update_sale
- delete_sale

규칙:

1. 등록 요청에는 create Tool을 사용하세요.
2. 수정 요청에는 update Tool을 사용하세요.
3. 삭제 요청에는 delete Tool을 사용하세요.
4. 필요한 데이터가 부족하면 임의로 생성하지 말고 사용자에게 요청하세요.
5. 기존 값을 유지하면서 일부 정보만 수정해야 한다면 먼저 get Tool로 현재 데이터를 확인하세요.
6. 사용자가 요청하지 않은 값은 임의로 변경하지 마세요.
7. 비밀번호는 사용자에게 출력하지 마세요.
8. Tool 실행 결과를 한국어로 설명하세요.
9. Tool 실행이 실패하면 성공했다고 답하지 마세요.
"""

class RouteDecision(BaseModel):
    route: Literal["sql", "mcp"] = Field(
        description=(
            "조회, 검색, 분석, 통계이면 sql / "
            "등록, 추가, 수정, 변경, 삭제이면 mcp"
        )
    )

router_llm = llm.with_structured_output(
    RouteDecision
)

class DBState(TypedDict):
    query: str
    route: str
    result: str

async def router_node(state: DBState):

    prompt = f"""
다음 사용자 요청을 분류하세요.

분류 기준:

sql:
- 데이터 조회
- 검색
- 목록 확인
- 통계
- 합계
- 평균
- 비교
- 분석

mcp:
- 새로운 데이터 등록
- 추가
- 수정
- 변경
- 삭제

사용자 요청:
{state["query"]}
"""

    decision = await router_llm.ainvoke(prompt)

    return {
        "route": decision.route
    }


async def sql_node(state: DBState):

    response = await sql_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["query"]
                }
            ]
        }
    )

    answer = response["messages"][-1].content

    return {
        "result": answer
    }

async def create_mcp_agent():

    client = MultiServerMCPClient(
        {
            "database": {
                "transport": "stdio",
                "command": PYTHON_PATH,
                "args": [MCP_SERVER_PATH],
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=MCP_SYSTEM_PROMPT,
    )

    return agent

def create_db_graph(mcp_agent):

    async def mcp_node(state: DBState):

        response = await mcp_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": state["query"]
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        return {
            "result": answer
        }

    builder = StateGraph(DBState)

    # Node
    builder.add_node(
        "router",
        router_node
    )

    builder.add_node(
        "sql",
        sql_node
    )

    builder.add_node(
        "mcp",
        mcp_node
    )

    # START → Router
    builder.add_edge(
        START,
        "router"
    )

    # Router → SQL 또는 MCP
    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "sql": "sql",
            "mcp": "mcp"
        }
    )

    builder.add_edge(
        "sql",
        END
    )

    builder.add_edge(
        "mcp",
        END
    )

    return builder.compile()

async def ask_db_agent(graph, question: str):

    result = await graph.ainvoke(
        {
            "query": question,
            "route": "",
            "result": "",
        }
    )

    print("\n" + "=" * 70)

    print("[질문]")
    print(question)

    print("\n[Route]")
    print(result["route"])

    print("\n[답변]")
    print(result["result"])

    print("=" * 70)

async def main():

    print("DB Agent 초기화 중...")

    mcp_agent = await create_mcp_agent()

    graph = create_db_graph(
        mcp_agent
    )

    print("DB Agent 준비 완료")

    # ----------------------------------------
    # SQL Agent 테스트
    # ----------------------------------------

    await ask_db_agent(
        graph,
        "서울에 거주하는 회원은 몇 명이야?"
    )

    # ----------------------------------------
    # MCP Agent 테스트
    # ----------------------------------------

    await ask_db_agent(
        graph,
        """
새로운 회원을 등록해줘.

비밀번호: pw006
이름: 홍길동
성별: 남
연락처: 010-6666-6666
지역: 부산
비고: 신규회원
"""
    )

async def run_db_agent(
    graph,
    question: str
) -> str:

    result = await graph.ainvoke(
        {
            "query": question,
            "route": "",
            "result": "",
        }
    )

    return result["result"]

if __name__ == "__main__":
    asyncio.run(main())

