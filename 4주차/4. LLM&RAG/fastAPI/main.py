from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import uvicorn

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# 환경설정

# HTML에 사용하는 스타일, 스크립트, 이미지 등을 사용하기 위한 폴더 등록
# app.mount(URL경로, StaticFiles(directory=실제경로), name='static')
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')

# HTML 파일(브라우저에 출력되는 파일)
templates = Jinja2Templates(directory=BASE_DIR / 'templates')


# 루트(/) 접속 => 코드에서 HTML 생성

# @app.get('/', response_class=HTMLResponse)
# async def response_main(request:Request):
#   html_main = '''
#     <html>
#       <head>
#       <title>HTMLResponse</title>
#       </head>
#       <body>
#         <h2>Html Respones 출력 연습</h2>
#         <div>
#           <a href="/createitem">GoTo Create Form</a>
#         </div>
#       </body>
#     </html>
#     '''
#   return HTMLResponse(html_main, status_code=status.HTTP_200_OK)

# 루트(/) 접속 => /templates/index.html 파일 전달
@app.get('/', response_class=HTMLResponse)
async def response_main(request:Request):
  return templates.TemplateResponse(request=request, name='index.html')

# http://127.0.0.1:8080/createitem 로 요청하면 => 입력폼(createform.html) 전달
@app.get('/createitem',  response_class=HTMLResponse)
async def create_form(request:Request):
  return templates.TemplateResponse(request=request, name='createform.html')

# http://127.0.0.1:8080/inputredirect => 데이터 전송 및 처리
@app.post('/inputredirect')
async def create_item(request:Request,
                      itemid: int = Form(...),
                      itemname: str = Form(...)):
  
  # 입력 데이터 서버(터미널) 화면에 출력
  print(f'itemid: {itemid}, itemname: {itemname}')

  # 접속 URL 생성
  target = request.url_for('itemview', itemid = itemid)
  # => 생성된 url => '/itemview/{itemid}'

  target = target.include_query_params(itemname=itemname)
  # => 생성된 url => '/itemview/{itemid}?itemname={itemname}
  print(target)

  # 새로 생성한 URL로 데이터 전달하기
  # POST => GET 으로 RedirectRespons를 이용해 변경(303 코드)
  return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)

# 출력을 위한 데이터 클래스 생성
class ResItem(BaseModel):
  id: int
  name: str | None = None

# 변경된 URL을 이용해 데이터 출력
@app.get('/itemview/{itemid}', name='itemview', response_class=HTMLResponse)
async def res_html(request:Request,
                   itemid: int,
                   itemname: str|None=None):
  
  item = ResItem(id = itemid,
                 name = itemname)
  
  item_dict = item.model_dump()
  print(item_dict)

  return templates.TemplateResponse(request=request,
                                    name='itemview.html',
                                    context={'item_dict':item_dict})


if __name__ == '__main__':
  uvicorn.run('main:app', host='127.0.0.1', port=8080, reload=True)

  # 프롬프트(터미널) 실행 명령
  # python -m uvicorn main:app --port 8080 --reload


