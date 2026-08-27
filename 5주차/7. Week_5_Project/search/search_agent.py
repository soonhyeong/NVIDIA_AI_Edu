import os
import asyncio

from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# ============================================================
# 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
    include_answer=False,
    include_raw_content=False,
)

result = tavily_search.invoke(
    {
        "query": "최근 NVIDIA AI 관련 주요 뉴스"
    }
)

SEARCH_SYSTEM_PROMPT = """
당신은 웹 검색을 활용해 최신 정보를 제공하는 Search Agent입니다.

사용자의 질문에 최신 정보나 외부 정보가 필요하면 Tavily Search Tool을 사용하세요.

규칙:

1. 최신 정보가 필요한 질문은 반드시 Tavily Search를 사용하세요.
2. 검색 결과에 없는 내용을 임의로 추측하지 마세요.
3. 검색 결과 여러 개를 비교하여 핵심 내용을 정리하세요.
4. 동일한 내용이 반복되면 하나로 정리하세요.
5. 서로 다른 출처의 내용이 충돌하면 차이를 설명하세요.
6. 가능하면 답변에 주요 출처의 제목과 URL을 함께 포함하세요.
7. 검색 결과를 그대로 복사하지 말고 사용자가 이해하기 쉽게 요약하세요.
8. 최종 답변은 한국어로 작성하세요.
"""

search_agent = create_agent(
    model=llm,
    tools=[tavily_search],
    system_prompt=SEARCH_SYSTEM_PROMPT,
)

async def ask_search_agent(question: str):

    result = await search_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\n" + "=" * 70)

    print("[질문]")
    print(question)

    print("\n[Tool 호출]")

    for message in result["messages"]:

        if hasattr(message, "tool_calls") and message.tool_calls:

            for tool_call in message.tool_calls:

                print("Tool:", tool_call["name"])
                print("Args:", tool_call["args"])

    print("\n[답변]")
    print(result["messages"][-1].content)

    print("=" * 70)

    return result

async def main():

    print("Search Agent 준비 완료")

    await ask_search_agent(
        "최근 NVIDIA AI 관련 주요 뉴스를 알려줘."
    )

    await ask_search_agent(
        "최근 LangGraph 주요 업데이트를 알려줘."
    )

    await ask_search_agent(
        "현재 생성형 AI 시장의 주요 트렌드를 알려줘."
    )

async def run_search_agent(
    question: str
) -> str:

    result = await search_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return result["messages"][-1].content

if __name__ == "__main__":
    asyncio.run(main())

