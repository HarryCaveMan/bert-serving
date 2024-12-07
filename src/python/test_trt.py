import asyncio
from trt_model import TRTModel

async def main():
    model =TRTModel("sentence-transformers/all-MiniLM-L12-v2","sentence_transformer_model.trt",io_buffer_sets_per_stream=1)
    print(await model.predict(["Hello, World!"]))
    

asyncio.run(main())