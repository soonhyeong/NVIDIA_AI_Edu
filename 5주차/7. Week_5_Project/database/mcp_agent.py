import os
import asyncio

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

PYTHON_PATH = r"C:\miniconda\envs\langchain_up\python.exe"

MCP_SERVER_PATH = str(
    BASE_DIR / "database" / "mcp_server.py"
)

LLM_MODEL = os.getenv("OPENAI_MODEL")


# ============================================================
# MCP Agent Prompt
# ============================================================

MCP_SYSTEM_PROMPT = """
당신은 회원 및 판매 데이터를 관리하는 데이터베이스 관리 Agent입니다.

사용자는 자연어로 회원 및 판매 데이터의 조회, 등록, 수정, 삭제를 요청합니다.

사용 가능한 Tool:

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

1. 사용자의 요청에 맞는 MCP Tool을 선택하세요.
2. 데이터베이스의 값을 임의로 추측하지 마세요.
3. Tool 실행에 필요한 정보가 부족하면 사용자에게 추가 정보를 요청하세요.
4. Tool 실행 결과를 이해하기 쉬운 한국어로 설명하세요.
5. Tool 실행에 실패하면 성공했다고 답하지 마세요.
6. 비밀번호 정보는 사용자에게 출력하지 마세요.
7. 사용자가 요청하지 않은 데이터를 수정하거나 삭제하지 마세요.
"""


# ============================================================
# MCP Agent 실행
# ============================================================

async def main():

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # MCP Client
    client = MultiServerMCPClient(
        {
            "database": {
                "transport": "stdio",
                "command": PYTHON_PATH,
                "args": [MCP_SERVER_PATH],
            }
        }
    )

    # MCP Tool 불러오기
    tools = await client.get_tools()

    print("MCP 연결 성공")
    print(f"등록된 Tool 수: {len(tools)}")

    for tool in tools:
        print(f"- {tool.name}")

    # Agent 생성
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=MCP_SYSTEM_PROMPT,
    )

    # 첫 번째 테스트
    question = "1번 회원 정보를 알려줘."

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    # 사용한 Tool 확인
    print("\n[Tool 호출]")

    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                print("Tool:", tool_call["name"])
                print("Args:", tool_call["args"])

    # 최종 응답
    print("\n[최종 답변]")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())