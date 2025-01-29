from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from trt_model import TRTModel

class Request(BaseModel):
    texts: List[str]

class Response(BaseModel):
    embeddings: List[List[float]]


class TRTService:
    def __init__(
        self,
        model_path,
        trt_model,
        num_contexts=1,
        num_cuda_streams_per_context=1,
        io_buffer_sets_per_stream=4
    ) -> None:
        self.app = FastAPI()
        self.router = APIRouter()
        self.model_path = model_path
        self.trt_model = trt_model
        self.num_contexts = num_contexts
        self.num_cuda_streams_per_context = num_cuda_streams_per_context
        self.io_buffer_sets_per_stream = io_buffer_sets_per_stream
        self.app.add_event_handler("startup", self.wait_model_ready)
        self.router.add_api_route("/encode", self.encode, methods=["POST"])
        self.app.include_router(self.router)
        

    async def wait_model_ready(self,timeout=60):
        self.model = TRTModel(
            self.model_path,
            self.trt_model,
            num_contexts=self.num_contexts,
            num_cuda_streams_per_context=self.num_cuda_streams_per_context,
            io_buffer_sets_per_stream=self.io_buffer_sets_per_stream
        )
    
    async def encode(self,request: Request) -> Response:
        model_outputs = await self.model.predict(request.texts)
        return Response(embeddings=model_outputs["sentence_embeddings"].tolist())

if __name__ == "__main__":
    service = TRTService(
        model_path="sentence-transformers/all-MiniLM-L12-v2",
        trt_model="model_10_7.trt",
        num_contexts=1,
        num_cuda_streams_per_context=6,
        io_buffer_sets_per_stream=6
    )
    uvicorn.run(service.app, host="0.0.0.0", port=8080)