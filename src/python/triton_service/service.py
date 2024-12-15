from triton_model import TritonServerWithAioClient
from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
from typing import List,Optional
from functools import cached_property
from pathlib import Path
from os import environ as env
import uvicorn
import asyncio
import logging
import time

class Request(BaseModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    texts: List[str]

class Response(BaseModel):
    embeddings: List[List[float]]

class TritonService:

    def __init__(
        self, 
        model_repository: str,
        model_name: str = "all-MiniLM-L12-v2",
        model_version: str = "1",
        log_dir: str = "/opt/service/logs",
        log_level: str = "INFO"
    ):
        Path(log_dir).mkdir(parents=True,exist_ok=True)
        logging.basicConfig(
            filename=f"{log_dir}/app.log",
            level=log_level
        )
        self.model_repository = model_repository
        self.model_name = model_name
        self.model_version = model_version
        self.app = FastAPI()
        self.router = APIRouter()
        self.router.add_api_route("/encode", self.encode, methods=["POST"])
        self.app.include_router(self.router)
        # Register the startup and shutdown events
        self.app.add_event_handler("startup", self.wait_model_ready)
        self.app.add_event_handler("shutdown", self.shutdown)
        self.log_dir = log_dir
        self.log_level = log_level

    @cached_property
    def logger(self):
        return logging.getLogger("app")

    async def wait_model_ready(self,timeout=60):
        self.logger.info("Starting Triton server...")
        self.model = TritonServerWithAioClient(
            self.model_repository,
            self.model_name,
            self.model_version,
            self.log_dir
        )
        await self.model.start()
        self.logger.info("Triton server {}".format("started" if self.model.ready else "failed to start"))
        if not self.model.ready:
            await self.shutdown()
            raise Exception("Triton server failed to start")

    async def encode(self, request: Request) -> Response:
        model_outputs = await self.model.predict(request.texts)
        self.logger.debug(f"Model outputs: {model_outputs}")
        return Response(embeddings=model_outputs["sentence_embeddings"].tolist())

    async def shutdown(self):
        if hasattr(self,"model"):
            self.logger.info("Stopping Triton server...")
            await self.model.stop()
            self.logger.info("Done")


if __name__ == "__main__":
    log_level = env.get("LOG_LEVEL","INFO")
    service = TritonService("triton",log_level=log_level)
    uvicorn.run(service.app, host="0.0.0.0", port=8080)