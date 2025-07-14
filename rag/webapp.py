import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path

from scripts.parse_excel import parse_excel_file
from scripts.parse_pdf import parse_pdf_to_chunks
from rag.retriever import EnhancedRetriever
from rag.chatbot import RAGChatbot
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

retriever = EnhancedRetriever()

# Gemma API URL 설정
gemma_api_url = os.getenv('GEMMA_API_URL')
if gemma_api_url:
    print("✅ Gemma API URL이 설정되어 있습니다. Gemma 모델을 사용합니다.")
    chatbot = RAGChatbot(
        retriever,
        gemma_api_url=gemma_api_url,
        gemma_model="google/gemma-3-12b-it"
    )
else:
    print("⚠️ Gemma API URL이 설정되지 않았습니다. 대체 답변 모드를 사용합니다.")
    chatbot = RAGChatbot(retriever)

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if ext in [".xlsx", ".xls"]:
        chunks = parse_excel_file(str(save_path))
    elif ext == ".pdf":
        chunks = parse_pdf_to_chunks(str(save_path), output_path="temp.json")
    else:
        return JSONResponse({"error": "지원하지 않는 파일 형식"}, status_code=400)
    retriever.add_chunks(chunks)
    return {"result": "success", "chunks": len(chunks)}

@app.post("/chat")
async def chat_api(query: str = Form(...)):
    result = chatbot.chat(query)
    return {"response": result["response"]} 