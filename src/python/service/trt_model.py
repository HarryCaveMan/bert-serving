from transformers import AutoConfig,AutoTokenizer
import numpy as np
import tensorrt as trt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
from cuda_asyncio import (
    HostDeviceMem,
    IOBufferSet,
    allocate_buffers,
    CudaStream,
    CudaContext,
    CudaError
)

class TRTContextWithStreamAndBuffers:
    def __init__(self,engine,model_dim,num_buffer_sets) -> None:
        # Multiply by two because each buffer set contains:
        #   - one set of input buffers
        #   - one set of output buffers
        self._io_thread_pool = ThreadPoolExecutor(max_workers=num_buffer_sets)
        self._trt_context = engine.create_execution_context()
        self.stream = CudaStream()
        self._buffer_pool = tuple(allocate_buffers(self,model_dim) for _ in range(num_buffer_sets))

    def __del__(self):
        self._io_thread_pool.shutdown()

    def get_io_buffers(self) -> IOBufferSet:
        return next(
            # Try for next available
            (buffer_set for buffer_set in self._buffer_pool if buffer_set.idle),
            # default to least busy (fewest taints) if none are idle (The buffers have locking anyway)
            min(self._buffer_pool, key=lambda buffer_set: buffer_set.taints)
        )
    
    async def execute(self,**inputs) -> np.ndarray:
        with self.get_io_buffers() as buffers:
            print("start push")
            await buffers.push(**inputs)
            print("end push")
            print("start binding")
            output_shapes = {}
            for tensor_name,binding in buffers.bindings.items():
                self._trt_context.set_tensor_address(tensor_name,binding)
                if self._trt_context.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self._trt_context.set_input_shape(tensor_name, inputs[tensor_name].shape)
            print("end binding")
            print("start await execute")
            await buffers.async_exec(
                self._trt_context.execute_async_v3,
                self.stream.handle
            )
            print("end await execute")
            print("start await pull")
            output_shapes = {
                output_name:self._trt_context.get_tensor_shape(output_name) 
                for output_name in buffers.outputs.keys()
            }
            model_output = await buffers.pull(**output_shapes)
            print("end await pull")
        return model_output
  
    @property
    def idle(self):
        return self.stream.is_done()

    @property
    def taints(self) -> int:
        return sum(buffer_set.taints for buffer_set in self._buffer_pool)


class TRTExecPool:
    def __init__(self,engine,model_dim,num_cuda_streams,io_buffer_sets_per_stream) -> None:
        self._execution_contexts = tuple(
            TRTContextWithStreamAndBuffers(
                engine,
                model_dim,
                io_buffer_sets_per_stream
            )
            for i in range(num_cuda_streams)
        )
    
    def get_execution_context(self):
        return next(
            # Try for next available
            (stream for stream in self._execution_contexts if stream.idle),
            # default to least busy (fewest taints) if none are idle
            min(self._execution_contexts, key=lambda context: context.taints)
        )


class TRTModel:

    def load_engine(self, model_path) -> None:
        # loads the model from given filepath
        with open(model_path, "rb") as f:
            self._trt_engine = self._runtime.deserialize_cuda_engine(f.read())

    def load_tokenizer(self,model_path,trust_remote_code) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=trust_remote_code)
        self._hf_config = AutoConfig.from_pretrained(model_path,trust_remote_code=trust_remote_code)
        self._max_seq_len = self._hf_config.max_position_embeddings

    def __init__(self,model_path,trt_model,num_cuda_streams=1,io_buffer_sets_per_stream=4,**kwargs) -> None:
        trust_remote_code = kwargs.get("trust_remote_code",True)
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self.load_engine(str(Path(model_path)/"trt"/trt_model))
        self.load_tokenizer(model_path,trust_remote_code)
        self._exec_pool = TRTExecPool(
            self._trt_engine,
            self._hf_config.hidden_size,
            num_cuda_streams,
            io_buffer_sets_per_stream
        )

    async def predict(self,input_data):
        executor = self._exec_pool.get_execution_context()
        input_tensors = self._tokenizer(
            input_data,
            truncation=True,
            padding="longest",
            max_length=self._max_seq_len,
            return_tensors="np"
        )
        # for tensor_name in input_tensors:
        #     input_tensors[tensor_name] = np.vstack(input_tensors[tensor_name])
        return await executor.execute(**input_tensors)