import uvicorn
from fastapi import FastAPI
from backend.routers import api  # 引入接口模块

app = FastAPI(title="房价数据分析系统 API")

# 挂载接口路由
app.include_router(api.router)

def main():
    print("=== 房价数据分析系统 API 启动中 ===")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

