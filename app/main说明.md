# -*- coding: utf-8 -*-
"""
FastAPI 应用入口：生命周期内初始化异步数据库引擎。
create fastapi app
"""

from contextlib import asynccontextmanager #异步上下文管理器

from fastapi import FastAPI 
from loguru import logger #封装好的日志库

from app.api.routes import chat, document, health #三个路由，聊天，文档，健康检查
from app.config import get_settings #自定义的配置获取函数


'''init_engine：初始化异步数据库引擎（建立和数据库的底层连接）
configure_session：配置数据库会话工厂，绑定引擎，给后续业务接口提供数据库会话。
'''
from app.infrastructure.database.session import configure_session, init_engine

# 设计思路：利用 FastAPI 的 lifespan 生命周期事件，在应用启动时创建数据库引擎并绑定到 app.state，在应用关闭时清理资源。
'''装饰器'''
@asynccontextmanager
async def lifespan(app: FastAPI):
    #调用项目配置函数，一次性拿到：项目名、运行环境、数据库地址、接口前缀等所有配置。
    settings = get_settings()
    #输出日志：应用名，运行环境
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    #初始化数据库引擎，建立连接池等底层资源准备工作。（连接数据库）
    engine = init_engine(settings.database_url)
    configure_session(engine)
    '''把数据库引擎绑定到 app.state，供后续业务接口使用（如依赖注入获取数据库会话）。'''
    app.state.engine = engine
    yield #分水领，上面是启动前准备，下面是关闭前清理，当服务收到关闭信号时，从 yield 往下继续执行
    #关闭数据库引擎，释放连接池等资源。
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


'''这是 FastAPI 应用工厂函数，专门负责：
读取配置 → 创建 FastAPI 实例 → 绑定生命周期 → 
注册所有业务路由 → 返回完整应用对象'''

def create_app() -> FastAPI:
    settings = get_settings()
    #创建 FastAPI 应用实例，绑定生命周期事件（lifespan），注册路由（健康检查、聊天、文档），最后返回应用对象。
    application = FastAPI(
        title=settings.app_name,#设置接口文档（Swagger）页面的标题，显示项目名称
        debug=settings.debug,#是否开启调试模式：开发环境开启、生产环境关闭
        lifespan=lifespan,#绑定生命周期事件，应用启动时执行 lifespan 函数的前半部分，应用关闭时执行后半部分，实现数据库引擎的自动管理。
    )

include_router()：FastAPI 挂载子路由的固定方法
把拆分好的三个业务模块路由注册进主应用：
health：服务健康检查接口
chat：聊天业务接口
document：文档业务接口
prefix=settings.api_prefix：统一全局接口前缀
比如配置是 /api/v1，那所有接口都会自动带上这个前缀，不用每个路由单独写。

    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    return application


app = create_app()
