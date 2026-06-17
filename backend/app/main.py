from fastapi import FastAPI

from app.routes import collections

app = FastAPI()

app.include_router(collections.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}