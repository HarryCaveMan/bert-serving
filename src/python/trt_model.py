from transformers import AutoConfig,AutoTokenizer
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from itertools import cycle

class HostDeviceMem:
    def __init__(self, host_mem, device_mem, shape):
        # keeping track of addresses
        self.host = host_mem
        self.device = device_mem
        # keeping track of shape to un-flatten it later
        self.shape = shape


class IOBufferSet:
    def __init__(self,inputs,outputs,bindings,stream,thread_pool) -> None:
        self._thread_pool = thread_pool
        self._loop = asyncio.get_running_loop()
        self._input_guard = asyncio.Lock()
        self._output_guard = asyncio.Lock()
        self.inputs = inputs
        self.output = outputs
        self.bindings = bindings
        self.stream = stream
        self._taints = 0

    def taint(self) -> None:
        self._taints += 1

    def clean(self) -> None:
        self._taints -= 1

    @property
    def idle(self) -> bool:
        return self._taints == 0
        
    # Awaits a cuda stream event asynchronously without blocking the main event loop
    # Currently each buffer gets its own thread because The streams I/O buffers lock independently of each other
    # BUffers only lock when dependent on the HOST (output)buffer contents or are mutating the HOST (input)buffer 
    async def sync_cuda_event(self,event) -> None:
        await self._loop.run_in_executor(
            self._thread_pool,
            self.stream.wait_for_event,
            event
        )

    def dtoh_async(self) -> None:
        await buffers.begin_read()
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output.host, 
                output.device, 
                self.stream
            )

    def htod_async(self,input_index) -> None:
        cuda.memcpy_htod_async(
            self.inputs[input_index].device, 
            self.inputs[input_index].host, 
            self.stream
        )

    async def push_one(self,input_data,input_index=0) -> None:
        completion_event = cuda.Event()
        await self._input_guard.acquire()
        try:
            np.copyto(self.inputs[input_index].host, input_data.ravel())
            self.htod_async(input_index)
            self.stream.record(completion_event)
            await self.sync_cuda_event(completion_event)
        finally:
            self._input_guard.release()        

    async def pull(self,event) -> np.ndarray:
        await self._output_guard.acquire()
        self.dtoh_async()
        event.record(self.stream)
        await self.sync_cuda_event(event)
        outputs = []
        try:
            for host_device_output in self.outputs:
                output = np.copy(host_device_output.host)  # This is a NumPy ndarray
                output = output.reshape(host_device_output.shape)  # Reshape to original tensor shape
                outputs.append(output)            
            return outputs
        finally:
            self._output_guard.release() # Can free output Buffer for reuse after deepcopy
    

class TRTContextWithStreamAndBuffers:
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
            # default to round-robin if none are idle (The buffers have locking anyway)
            next(cycle(self._buffer_pool))
        )
    
    async def execute(self,input_data) -> np.ndarray:
        completion_event = cuda.Event()
        buffers = self.get_io_buffers()
        buffers.taint()
        await buffers.push_one(input_data)
        self._trt_context.enqueue_v3(
            buffers.stream.handle,
            buffers.bindings
        )
        model_output = await buffers.pull(completion_event)
        buffers.clean()
        return model_output
  
    @property
    def idle(self):
        return not self.stream.is_done()

    def __init__(self,engine,num_buffers) -> None:
        self._io_thread_pool = ThreadPoolExecutor(max_workers=num_buffers*2)
        self._input_guard = asyncio.Lock()
        self._output_guard = asyncio.Lock()
        self._trt_context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self._buffer_pool = tuple((self.allocate_buffers() for i in range(num_buffers)))


class TRTExecPool:
    def __init__(self,engine,num_cuda_streams,num_buffers_per_stream) -> None:
        self._guard = asyncio.Semaphore(num_cuda_streams)
        self._execution_contexts = (
            TRTContextWithStreamAndBuffers(
                engine,
                num_buffers_per_stream
            )
            for i in range(num_cuda_streams)
        )
    
    def get_execution_context(self):
        return next(
            # Try for next available
            (stream for stream in self._execution_contexts if stream.idle),
            # default to round-robin if none are idle
            next(cycle(self._execution_contexts))
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