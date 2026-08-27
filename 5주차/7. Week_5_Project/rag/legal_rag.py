import os

from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 환경 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_DB_DIR = (
    BASE_DIR
    / "rag"
    / "chroma_db"
)


# ============================================================
# Embedding
# ============================================================

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# 기존 Chroma DB 로드
# ============================================================

vectorstore = Chroma(
    collection_name="legal_rag",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DB_DIR),
)


# ============================================================
# Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)


# ============================================================
# Prompt
# ============================================================

LEGAL_SYSTEM_PROMPT = """
당신은 대한민국 법률 문서를 기반으로 답변하는 법률 정보 도우미입니다.

규칙:

1. 제공된 법률 문서를 근거로 답변하세요.
2. 문서에서 확인되지 않는 내용을 임의로 추측하지 마세요.
3. 답변의 근거가 되는 법률명과 조문을 명시하세요.
4. 질문과 직접 관련된 법률을 우선해서 설명하세요.
5. 법률의 의미를 임의로 변경하지 마세요.
6. 근거가 부족하면
   "제공된 법률 문서에서 확인하기 어렵습니다."
   라고 답하세요.

[법률 문서]
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            LEGAL_SYSTEM_PROMPT
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ============================================================
# 검색 결과 Formatting
# ============================================================

def format_documents(
    documents: list[Document]
) -> str:

    formatted = []

    for doc in documents:

        law_name = doc.metadata.get(
            "law_name",
            "법률명 없음"
        )

        article = doc.metadata.get(
            "article",
            "조문 없음"
        )

        formatted.append(
            f"""
[{law_name} {article}]
{doc.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(formatted)


# ============================================================
# 법률 RAG 실행 함수
# ============================================================

async def ask_legal_rag(
    question: str
) -> str:

    documents = await retriever.ainvoke(
        question
    )

    context = format_documents(
        documents
    )

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = await llm.ainvoke(
        messages
    )

    return response.content