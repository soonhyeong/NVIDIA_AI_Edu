from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI() # FastAPI 객체 생성

# 가상DB 생성
items_db = [
    {"item_name": "Item 1", "price": 1000},
    {"item_name": "Item 2", "price": 2000},
    {"item_name": "Item 3", "price": 3000},
    {"item_name": "Item 4", "price": 4000}
]

#  ===========================================================

@app.get("/") # GET 요청 처리
async def root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}") # 경로 매개변수 처리
async def item(item_id: int): # 쿼리 매개변수 처리
    return {"item_id": item_id}

# items_db 전체 조회
@app.get("/item")
async def item_fun():
    return{"items": items_db}

# items_db 특정 아이템 조회
@app.get("/item/{item_id}")
async def get_item(item_id: int):
    return {"item": items_db[item_id]}


if __name__ == "__main__":
    uvicorn.run('main_app:app', 
    
                host="127.0.0.1", 
                port=8000,
                reload=True) # 코드 변경 시 자동 재시작