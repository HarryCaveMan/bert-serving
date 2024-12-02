from transformers import AutoConfig,AutoTokenizer
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import asyncio
from pathlib import Path
from itertools import cycle

class HostDeviceMem:
    def __init__(self, host_mem, device_mem, shape):
        # keeping track of addresses
        self.host = host_mem
        self.device = device_mem
        # keeping track of shape to un-flatten it later
        self.shape = shape


class TRTContextWithStreamAndBuffers:
    def allocate_buffers(self):
    """Allocate buffers dynamically based on the engine's context and current input shape."""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding_index in range(self._trt_context.engine.num_bindings):
        binding_name = self._trt_context.engine.get_binding_name(binding_index)
        shape = self._trt_context.get_binding_shape(binding_name)
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
    return inputs, outputs, bindings, stream

    async def attach(self) -> None:
        await self._lock.acquire()
        self._in_use = True

    def detach(self) -> None:
        self._lock.release()
        self._in_use = False

    def dtoh_async(self) -> None:
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output.host, 
                output.device, 
                self.stream
            )

    def push_async(self,input_data,input_index=0) -> None:
            np.copyto(_input.host, input_data.ravel())
            cuda.memcpy_htod_async(
                self.inputs[input_index].device, 
                self.inputs[input_index].host, 
                self.stream
            )

    async def pull(self,event) -> np.ndarray:
        await self._loop.run_in_executor(
            None,
            self.stream.wait_for_event,
            event
        )
        outputs = []
        for host_device_output in self.outputs:
            output = host_device_output.host  # This is a NumPy ndarray
            output = output.reshape(host_device_output.shape)  # Reshape to original tensor shape
            outputs.append(output)
        return outputs

    async def execute(self,input_data) -> np.ndarray:
        completion_event = cuda.Event()
        self.push_async(input_data)
        self._trt_context.execute_async_v3(self.stream.handle)
        self.dtoh_async()
        completion_event.record(self.stream)
        return await self.pull(completion_event)
  
    @property
    def available(self):
        return not self._in_use

    def __init__(self,engine) -> None:
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        self._trt_context = engine.create_execution_context()
        self.inputs,self.outputs,self.bindings,self.stream = self.allocate_buffers(engine)
        self.detach()


class TRTExecPool:
    def __init__(self,engine,max_concurrency,logger) -> None:
        self._guard = asyncio.Semaphore(max_concurrency)
        self._execution_contexts = (
            TRTContextWithStreamAndBuffers(engine)
            for i in range(max_concurrency)
        )
    
    async def get_execution_context(self):
        await self._guard.acquire()
        try:
            context = next(
                # Try for next available
                filter(
                    lambda stream:stream.available,
                    self._execution_contexts
                ),
                # default to round-robin if none available
                # await below will still wait until the context lock is released
               next(cycle(self._execution_contexts))
            )
            await context.attach()
            return context
        except Exception as e:
            try: context.detach()
            # just pass if exceptioin was thrown before stream attach
            except Exception: pass
            self._guard.release()
            raise e

    def release_stream(self,stream):
        try:
            stream.detach()
        finally:
            self._guard.release()


class TRTModel:
    def load_engine(self, model_path) -> None:
        # loads the model from given filepath
        with open(Path(engine_path)/"model.trt", "rb") as f:
            self._trt_engine = self._runtime.deserialize_cuda_engine(f.read())

    def load_tokenizer(self,model_path) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._hf_config = AutoConfig.from_pretrained(model_path)
        self._max_seq_len = self._hf_config.max_position_embeddings

    def __init__(self,model_path,max_concurrency=1) -> None:
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self.logger)
        self.load_engine(model_path)
        self.load_tokenizer(model_path)
        self._exec_pool = TRTExecPool(self._trt_engine,max_concurrency)

    async def predict(self,input_data):
        executor = await self._exec_pool.get_execution_context()
        inputs_np = self._tokenizer.tokenize(
            input_data,
            truncation=True,
            max_length=self._max_seq_len,
            output_tensors="np"
        )

        