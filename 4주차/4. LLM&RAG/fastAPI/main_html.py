# 모듈 연결
from fastapi import FastAPI, Form, Request     
from pydantic import BaseModel   
from fastapi.responses import HTMLResponse # 데이터를 html에 전달
from fastapi.templating import Jinja2Templates  # html 화면을 렌더링 할때 사용하는 도구
from urllib3 import request
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='templates')  # 데이터를 전달할 html 파일이 있는 폴더

# 데이터 모델 선언(재사용)
class Item(BaseModel):
  name: str
  description: str | None=None
  price: float
  tax: float | None=None

# form.html 파일의 <form> 태그를 이용한 값 요청(get)
@app.get('/', response_class=HTMLResponse)
async def show_form(request:Request):
  return templates.TemplateResponse(
   request=request,
    name="form.html",
  ) 
@app.post('/items_tax/submit', response_class=HTMLResponse)
async def submit_form(request:Request,    # 클라이언트(브라우저/앱)가 보낸 HTML 요청 정보 전체를 담는 객체
                      name: str = Form(...), 
                      # Form(...): html의 form 테그에서 온 데이터를 바로 받을 수 있음(name 속성 사용)
                      description: str | None=Form(None),
                      price: float = Form(...),
                      tax: float | None=Form(None)):
  
  # 폼에서 받아온 데이터를 pydantic 모델로 변환
  item = Item(name=name,
              description=description,
              price=price,
              tax=tax)
  
  item_dict = item.model_dump()  # 딕셔너리 형식으로 변환(입력값을 클라이언트에게 반환)

  # tax에 따를 총금액 계산
  if item.tax:
    price_with_tax = item.price + item.tax
    item_dict.update({'price_with_tax': price_with_tax})

  # return item_dict  # 백에서 프론트로 전송시 사용하는 방법

  # 요청한 클라이언트(브라우저/앱)에게 입력값과 계산 결과 전달
  return templates.TemplateResponse(
    request=request,
    name="result.html",
    context={"item": item_dict},
)
  


# uvicorn을 이용한 내부 실행
if __name__ == '__main__':  # 현재 파일에서 실행 요청 확인
  uvicorn.run('main_html:app',   # 실행할 파일과 대상 앱
              host='127.0.0.1',  # 실행시 도메인 주소
              port=8081,   # 접속 포트(기본포트:8000)
              reload=True) # 코드 변경시 자동 적용후 재실행

# python -m uvicorn main_html:app --port 8081 --reload