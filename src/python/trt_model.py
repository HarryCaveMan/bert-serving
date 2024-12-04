from transformers import AutoConfig,AutoTokenizer
import numpy as np
import tensorrt as trt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio

from cuda_asyncio import HostDeviceMem,IOBufferSet

class TRTContextWithStreamAndBuffers:
    def __init__(self,engine,num_buffer_sets) -> None:
        # Multiply by two because each buffer set contains:
        #   - one set of input buffers
        #   - one set of output buffers
        self._io_thread_pool = ThreadPoolExecutor(max_workers=num_buffer_sets*2)
        self._input_guard = asyncio.Lock()
        self._output_guard = asyncio.Lock()
        self._trt_context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self._buffer_pool = tuple(self.allocate_buffers() for i in range(num_buffer_sets))

    def allocate_buffers(self):
        inputs,outputs,bindings = [],[],[]
        for binding_index in range(self._trt_context.engine.num_bindings):
            binding_name = self._trt_context.engine.get_binding_name(binding_index)
            shape = self._trt_context.engine.get_binding_shape(binding_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self._trt_context.engine.get_binding_dtype(binding_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            mem_obj = HostDeviceMem(host_mem, device_mem, shape)
            if self._trt_context.engine.binding_is_input(binding_name):
                inputs.append(mem_obj)
            else:
                outputs.append(mem_obj)
        return IOBufferSet(inputs,outputs,bindings,self.stream,self._io_thread_pool)

    def get_io_buffers(self) -> IOBufferSet:
        return next(
            # Try for next available
            (buffer_set for buffer_set in self._buffer_pool if buffer_set.idle),
            # default to least busy (fewest taints) if none are idle (The buffers have locking anyway)
            min(self._buffer_pool, key=lambda buffer_set: buffer_set.taints)
        )
    
    async def execute(self,input_data) -> np.ndarray:
        completion_event = cuda.Event()
        buffers = self.get_io_buffers()
        buffers.taint()
        await buffers.push_one(input_data)
        self._trt_context.enqueue_v3(
            buffers.bindings,
            buffers.stream.handle
        )
        model_output = await buffers.pull(completion_event)
        buffers.clean()
        return model_output
  
    @property
    def idle(self):
        return not self.stream.is_done()

    @property
    def taints(self) -> int:
        return sum(buffer_set.taints for buffer_set in self._buffer_pool)


class TRTExecPool:
    def __init__(self,engine,num_cuda_streams,io_buffer_sets_per_stream) -> None:
        self._execution_contexts = tuple(
            TRTContextWithStreamAndBuffers(
                engine,
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
        with open(Path(model_path)/"model.trt", "rb") as f:
            self._trt_engine = self._runtime.deserialize_cuda_engine(f.read())

    def load_tokenizer(self,model_path) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._hf_config = AutoConfig.from_pretrained(model_path)
        self._max_seq_len = self._hf_config.max_position_embeddings

    def __init__(self,model_path,num_cuda_streams=1,io_buffer_sets_per_stream=4) -> None:
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self.load_engine(model_path)
        self.load_tokenizer(model_path)
        self._exec_pool = TRTExecPool(
            self._trt_engine,
            num_cuda_streams,
            io_buffer_sets_per_stream
        )

    async def predict(self,input_data):
        executor = self._exec_pool.get_execution_context()
        inputs_np = self._tokenizer(
            input_data,
            truncation=True,
            max_length=self._max_seq_len,
            return_tensors="np"
        )
        return await executor.execute(inputs_np["input_ids"])