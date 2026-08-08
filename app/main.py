"""FastAPI 入口：静态页面 + 对话接口 + mock API。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api import chat, mock

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="织语 WeaveTalk · 家纺客服导购一体化智能体")

app.include_router(chat.router, prefix="/api")
app.include_router(mock.router, prefix="/api/mock")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
