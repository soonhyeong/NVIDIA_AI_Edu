# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# LLM 관련 라이브러리
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import uvicorn

# env 파일 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# 권한 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index_name = 'my-tax-index'
vector_store = PineconeVectorStore(index_name=index_name,
                                   embedding=embeddings,
                                   pinecone_api_key=os.getenv("PINECONE_API_KEY"))

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0)

# 2. 페르소나 설정 (System Prompt)
system_prompt = """
당신은 대한민국의 AI 세무사로 소득세에 대한 전문가입니다.
사용자 질문에 대해서는 아래 [법령 근거]를 바탕으로 정중하고 명확하게 답변합니다.

[답변 원칙]
1. 반드시 제공된 [법령 근거] 안에서만 답변하며, 추측하지 마세요.
2. 답변 시작 시 "네, 소득세 관련하여 문의주셨군요. 해당 내용을 법령에 근거하여 설명해 드리겠습니다."와 같은 정중한 어조를 사용하세요.
3. 근거가 되는 조항(예: 소득세법 제1조)을 반드시 언급하세요.
4. 법령에 내용이 없을 경우 "죄송합니다만, 현재 제가 가진 자료에서는 해당 내용을 찾을 수 없습니다."라고 안내하세요.

[법령 근거]:
{context}
"""

# QA 프롬프트 설정
qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# 3. 메모리 설정
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True, 
    output_key="answer"
)

# 4. 체인 설정
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

class TaxRequest(BaseModel):
    message: str

# 5. FastAPI 엔드포인트 설정
@app.post("/chat")
async def chat(request: TaxRequest):
    try:
        result = qa_chain.invoke({"question": request.message})
        return {"answer": result["answer"],
                "source": [doc.page_content for doc in result["source_documents"]]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러발생: {str(e)}")