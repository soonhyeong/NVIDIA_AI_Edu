"""소득세법 RAG 챗봇 백엔드.

기존 노트북의 DOCX 조문 분할, Ollama bge-m3 임베딩, Chroma 검색 구성을
재사용하면서 세액 계산 도구와 LangChain AgentExecutor를 완성한다.
"""

from __future__ import annotations

import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent
LAW_DOCUMENT_NAME = "소득세법_20260701.docx"


def _prefer_local_or_parent(local_path: Path, parent_path: Path) -> Path:
    """로컬 파일이 없으면 프로젝트 상위 폴더의 파일을 사용한다."""
    return local_path if local_path.is_file() else parent_path


LAW_FILE_PATH = _prefer_local_or_parent(
    PROJECT_DIR / "data" / LAW_DOCUMENT_NAME,
    PARENT_DIR / "data" / LAW_DOCUMENT_NAME,
)
ENV_FILE_PATH = _prefer_local_or_parent(
    PROJECT_DIR / ".env",
    PARENT_DIR / ".env",
)
LAW_DB_DIRECTORY = PROJECT_DIR / "chroma_income_tax_db"
COLLECTION_NAME = "income_tax_law_20260701"
EMBEDDING_MODEL = "bge-m3"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

load_dotenv(ENV_FILE_PATH)


class IncomeTaxInput(BaseModel):
    """종합소득 산출세액 계산 입력."""

    tax_base: int = Field(..., ge=0, description="종합소득 과세표준. 단위는 원.")


class LawSearchInput(BaseModel):
    """소득세법 검색 입력."""

    query: str = Field(..., min_length=2, description="찾을 조문 또는 세법 개념")


# 소득세법 제55조 제1항(2023년 이후 과세표준 구간)의 세율과 누진공제액.
TAX_BRACKETS = (
    (14_000_000, 6, 0),
    (50_000_000, 15, 1_260_000),
    (88_000_000, 24, 5_760_000),
    (150_000_000, 35, 15_440_000),
    (300_000_000, 38, 19_940_000),
    (500_000_000, 40, 25_940_000),
    (1_000_000_000, 42, 35_940_000),
    (None, 45, 65_940_000),
)


def calculate_income_tax_amount(tax_base: int) -> dict[str, int | str | None]:
    """과세표준에 대한 종합소득 산출세액을 결정적으로 계산한다."""
    if isinstance(tax_base, bool) or not isinstance(tax_base, int):
        raise TypeError("tax_base는 원 단위 정수여야 합니다.")
    if tax_base < 0:
        raise ValueError("tax_base는 0 이상이어야 합니다.")

    for upper_limit, rate, deduction in TAX_BRACKETS:
        if upper_limit is None or tax_base <= upper_limit:
            calculated_tax = max(0, tax_base * rate // 100 - deduction)
            return {
                "tax_base": tax_base,
                "upper_limit": upper_limit,
                "rate_percent": rate,
                "progressive_deduction": deduction,
                "calculated_tax": calculated_tax,
            }
    raise RuntimeError("세율 구간을 결정하지 못했습니다.")


@tool("calculate_income_tax", args_schema=IncomeTaxInput)
def calculate_income_tax(tax_base: int) -> str:
    """종합소득 과세표준(원)을 받아 소득세법 제55조의 산출세액을 계산합니다."""
    result = calculate_income_tax_amount(tax_base)
    upper = result["upper_limit"]
    bracket_text = (
        f"{upper:,}원 이하" if isinstance(upper, int) else "1,000,000,000원 초과"
    )
    return (
        "[종합소득 산출세액 계산 결과]\n"
        f"- 과세표준: {result['tax_base']:,}원\n"
        f"- 적용 구간: {bracket_text}\n"
        f"- 세율: {result['rate_percent']}%\n"
        f"- 누진공제액: {result['progressive_deduction']:,}원\n"
        f"- 산출세액: {result['calculated_tax']:,}원\n"
        "- 근거: 소득세법 제55조 제1항\n"
        "- 주의: 세액공제·세액감면·가산세·지방소득세는 반영하지 않았습니다."
    )


def split_law_articles(document: Document) -> list[Document]:
    """DOCX 본문을 제1조, 제1조의2 형태의 조문 단위로 분할한다."""
    text = document.page_content.replace("\r\n", "\n")
    pattern = re.compile(r"(?m)^\s*(제\d+조(?:의\d+)?(?:\([^\n)]+\))?)")
    matches = list(pattern.finditer(text))
    if not matches:
        return [document]

    articles: list[Document] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.start() : end].strip()
        heading = match.group(1)
        number_match = re.match(r"제\d+조(?:의\d+)?", heading)
        title_match = re.search(r"\(([^)]+)\)", heading)
        metadata = dict(document.metadata)
        metadata.update(
            {
                "article_number": number_match.group(0) if number_match else heading,
                "article_title": title_match.group(1) if title_match else "",
                "future_effective_note": "[시행일" in content,
            }
        )
        articles.append(Document(page_content=content, metadata=metadata))
    return articles


def load_and_split_law() -> list[Document]:
    """소득세법 DOCX를 읽고 긴 조문만 추가 청크로 나눈다."""
    if not LAW_FILE_PATH.is_file():
        raise FileNotFoundError(f"소득세법 문서를 찾을 수 없습니다: {LAW_FILE_PATH}")

    source_documents = Docx2txtLoader(str(LAW_FILE_PATH)).load()
    articles: list[Document] = []
    for document in source_documents:
        articles.extend(split_law_articles(document))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "①", "②", "③", "④", "⑤", ". ", " "],
    )
    return splitter.split_documents(articles)


def _embeddings() -> OllamaEmbeddings:
    kwargs: dict[str, str] = {"model": EMBEDDING_MODEL}
    if base_url := os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = base_url
    return OllamaEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """기존 Chroma 컬렉션을 재사용하고, 비어 있을 때만 최초 구축한다."""
    LAW_DB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    embeddings = _embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(LAW_DB_DIRECTORY),
        embedding_function=embeddings,
    )
    try:
        is_empty = vector_store._collection.count() == 0
    except Exception as exc:  # Chroma 연결/스키마 오류를 사용자 친화적으로 변환
        raise RuntimeError(f"Chroma DB를 열 수 없습니다: {exc}") from exc

    if is_empty:
        try:
            vector_store.add_documents(load_and_split_law())
        except Exception as exc:
            raise RuntimeError(
                "벡터 DB 최초 구축에 실패했습니다. Ollama가 실행 중이고 "
                f"'{EMBEDDING_MODEL}' 모델이 설치되어 있는지 확인하세요. 원인: {exc}"
            ) from exc
    return vector_store


def _exact_article_filter(query: str) -> dict[str, str] | None:
    match = re.search(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", query)
    if not match:
        return None
    number = f"제{match.group(1)}조"
    if match.group(2):
        number += f"의{match.group(2)}"
    return {"article_number": number}


@tool("search_income_tax_law", args_schema=LawSearchInput)
def search_income_tax_law(query: str) -> str:
    """2026년 7월 1일 시행 소득세법 DOCX에서 관련 조문과 근거를 검색합니다."""
    query = query.strip()
    if len(query) < 2:
        return "검색어를 두 글자 이상 입력해 주세요."
    try:
        store = get_vector_store()
        article_filter = _exact_article_filter(query)
        if article_filter:
            docs = store.similarity_search(query, k=5, filter=article_filter)
            # 이전에 만든 DB에 잘못 이어 붙은 청크가 있더라도, 정확한 조문
            # 검색에서는 해당 조문 제목으로 시작하는 원문 청크를 우선한다.
            expected = article_filter["article_number"]
            heading_docs = [
                doc
                for doc in docs
                if re.match(rf"^\s*{re.escape(expected)}(?:\(|\s)", doc.page_content)
            ]
            if heading_docs:
                docs = heading_docs
        else:
            docs = store.max_marginal_relevance_search(
                query, k=5, fetch_k=15, lambda_mult=0.7
            )
    except Exception as exc:
        return f"소득세법 검색 중 오류가 발생했습니다: {exc}"

    if not docs:
        return "첨부된 소득세법에서 관련 조문을 찾지 못했습니다. 검색어를 구체화해 주세요."

    results = []
    for index, doc in enumerate(docs, start=1):
        number = doc.metadata.get("article_number", "조문 번호 확인 불가")
        title = doc.metadata.get("article_title", "")
        future = "예" if doc.metadata.get("future_effective_note", False) else "아니요"
        results.append(
            f"[검색 결과 {index}]\n"
            f"조문: {number}{f' ({title})' if title else ''}\n"
            f"미래 시행 문구 포함: {future}\n"
            f"내용:\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(results)


TOOLS = [search_income_tax_law, calculate_income_tax]


def _few_shot_messages() -> list[BaseMessage]:
    return [
        HumanMessage(content="소득세법에서 거주자와 비거주자는 어떻게 구분하나요?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_income_tax_law",
                    "args": {"query": "거주자 비거주자 정의 제1조의2"},
                    "id": "law_example",
                }
            ],
        ),
        ToolMessage(
            content="제1조의2는 거주자와 비거주자의 정의를 규정한다.",
            tool_call_id="law_example",
        ),
        AIMessage(content="소득세법 제1조의2의 검색 결과를 근거로 구분해 설명합니다."),
        HumanMessage(content="종합소득 과세표준이 6천만원이면 산출세액은 얼마인가요?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_income_tax",
                    "args": {"tax_base": 60_000_000},
                    "id": "tax_example",
                }
            ],
        ),
        ToolMessage(
            content="과세표준 60,000,000원, 세율 24%, 누진공제 5,760,000원, 산출세액 8,640,000원. 근거 제55조.",
            tool_call_id="tax_example",
        ),
        AIMessage(content="산출세액은 8,640,000원이며 세액공제 등은 반영하지 않은 금액입니다."),
    ]


def build_prompt() -> ChatPromptTemplate:
    system = f"""당신은 대한민국 소득세법 질의응답 AI입니다.
기준 자료는 2026년 7월 1일 시행 소득세법 DOCX이며 오늘은 {date.today():%Y-%m-%d}입니다.

도구 사용 규칙:
0. 새 사용자 질문에 답할 때는 예시나 대화 기록에 같은 내용이 있어도 반드시 적절한 도구를 최소 한 번 호출한다.
1. 정의, 납세의무, 소득 구분, 비과세, 소득·세액공제, 세율 및 조문 질문은 search_income_tax_law를 사용한다.
2. 종합소득 과세표준이 주어지고 산출세액 계산을 요청하면 calculate_income_tax를 사용한다.
3. 계산 질문에서도 법적 근거가 필요하면 두 도구를 모두 사용할 수 있다.
4. 과세표준과 소득금액·총수입금액을 구분하고, 과세표준이 없으면 계산 전에 사용자에게 요청한다.
5. 검색 결과에 없는 내용이나 조문 번호를 만들지 않는다.
6. 답변에는 확인한 조문 번호와 핵심 근거를 명시한다.
7. 계산 결과에는 세액공제, 세액감면, 가산세, 지방소득세의 반영 여부를 명시한다.
8. 자료 기준일 이후 개정 여부는 이 챗봇만으로 확인할 수 없음을 필요할 때 알린다.
9. 답변은 명료한 한국어로 작성한다."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            *_few_shot_messages(),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


@lru_cache(maxsize=1)
def get_agent_executor() -> AgentExecutor:
    """실제 두 도구가 모두 등록된 Tool-calling AgentExecutor를 생성한다."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    agent = create_tool_calling_agent(llm, TOOLS, build_prompt())
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def ask_chatbot(
    question: str, chat_history: Sequence[BaseMessage] | None = None
) -> dict:
    """대화 기록과 질문을 AgentExecutor에 전달한다."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("질문을 입력해 주세요.")
    return get_agent_executor().invoke(
        {"input": cleaned, "chat_history": list(chat_history or [])}
    )


__all__ = [
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "LAW_DB_DIRECTORY",
    "LAW_FILE_PATH",
    "TOOLS",
    "IncomeTaxInput",
    "ask_chatbot",
    "calculate_income_tax",
    "calculate_income_tax_amount",
    "get_agent_executor",
    "get_vector_store",
    "search_income_tax_law",
    "split_law_articles",
]
