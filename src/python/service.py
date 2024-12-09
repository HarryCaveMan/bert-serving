from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from trt_model import TRTModel

class Request(BaseModel):
    texts: List[str]

class Response(BaseModel):
    embeddings: List[List[float]]

model = TRTModel("sentence-transformers/all-MiniLM-L12-v2","sentence_transformer_model.trt",io_buffer_sets_per_stream=4)

app = FastAPI()

@app.post("/encode")
async def encode(request: Request) -> Response:
    model_outputs = await model.predict(request.texts)
    return Response(embeddings=model_outputs["sentence_embeddings"].tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)