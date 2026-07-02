# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化异步数据库引擎。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.routes import agent, chat, document, health
from app.config import get_settings
from app.infrastructure.database.session import configure_session, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    engine = init_engine(settings.database_url)
    configure_session(engine)
    app.state.engine = engine
    yield
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    static_dir = Path(__file__).parent / "static"
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=static_dir), name="static")

        @application.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(agent.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    return application


app = create_app()
