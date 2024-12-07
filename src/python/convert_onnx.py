from trt_model import TRTModel

async def main():
    model = asyncio.run_coroutine_threadsafe(TRTModel("sentence-transformers/all-MiniLM-L12-v2","model.trt"),loop).result()
    print(await model.predict(["Hello, World!"]))

from convert_model import convert_hf_model
max_batch_size=64
optimal_batch_size=8
seq_len_optimization_factor=4
onnx_model ="model.onnx"
trt_model = "model.trt"
onnx_model
convert_hf_model(
    "sentence-transformers/all-MiniLM-L12-v2",
    onnx_model,
    trt_model,
    max_batch_size,
    optimal_batch_size,
    seq_len_optimization_factor
)