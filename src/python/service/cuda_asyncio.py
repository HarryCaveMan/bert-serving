import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import asyncio
from functools import cached_property

CudaStream = cuda.Stream
CudaEvent = cuda.Event
CudaContext = cuda.Context
CudaError = cuda.Error

def allocate_buffers(context_wrapper,model_dim):
    inputs,outputs,bindings = {},{},{}
    input_tensor_name = context_wrapper._trt_context.engine.get_tensor_name(0)
    profile_shape = context_wrapper._trt_context.engine.get_tensor_profile_shape(input_tensor_name, 0)
    # profile_shape contains (min_shape, opt_shape, max_shape)
    max_batch_size = profile_shape[2][0]  # First dim of max shape
    max_seq_len = profile_shape[2][1]
    for tensor_index in range(context_wrapper._trt_context.engine.num_io_tensors):
        tensor_name = context_wrapper._trt_context.engine.get_tensor_name(tensor_index)
        tensor_profile = context_wrapper._trt_context.engine.get_tensor_profile_shape(tensor_name,0)
        dtype = trt.nptype(context_wrapper._trt_context.engine.get_tensor_dtype(tensor_name))
        if context_wrapper._trt_context.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            max_shape = tensor_profile[2]
            mem = HostDeviceMem(max_shape,dtype)
            inputs[tensor_name] = mem
        else:
            max_shape = (max_batch_size,max_seq_len,model_dim)
            mem = HostDeviceMem(max_shape,dtype)
            outputs[tensor_name] = mem
        bindings[tensor_name] = mem.binding
    return IOBufferSet(
        inputs,
        outputs,
        bindings,
        context_wrapper.stream
    )

class HostDeviceMem:
    def __init__(self,max_shape,dtype):
        # Pagelocked host ptr and GPU(device) mem ptr
        self.host = cuda.pagelocked_empty(trt.volume(max_shape), dtype)
        self.device = cuda.mem_alloc(self.host.nbytes)
        # keeping track of shape to un-flatten it later
        self.binding = int(self.device)
    
    def __del__(self):
        self.host.base.free()
        self.device.free()


class IOBufferSet:
    def __init__(self,inputs,outputs,bindings,stream,asyncio_poll_interval_micros=0) -> None:
        self._input_guard = asyncio.Lock()
        self._output_guard = asyncio.Lock()
        self._htod_event = CudaEvent()
        self._dtoh_event = CudaEvent()
        self._exec_event = CudaEvent()
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.asyncio_poll_interval_micros = asyncio_poll_interval_micros
        self._taints = 0

    def __enter__(self):
        self._taints += 1
        return self

    def __exit__(self,exc_type,exc_val,exc_tb) -> None:
        self._taints -= 1

    @cached_property
    def _loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    @property
    def taints(self) -> int:
        return self._taints

    @property
    def idle(self) -> bool:
        return self._taints == 0
        
    # Awaits a cuda stream event asynchronously without blocking the main event loop
    # Currently each buffer gets its own thread because The streams I/O buffers lock independently of each other
    # CUDA streams handle synchronization of the accelerator, so locks only needed for HOST buffers
    # Buffers only lock when dependent on the HOST (output)buffer contents or while mutating the HOST (input)buffer
    async def sync_cuda_event_or_stream(self,event_or_stream,) -> None:
        while True:
            try:
                if event_or_stream.query():
                    break
            finally:
                await asyncio.sleep(self.asyncio_poll_interval_micros/1e6)

    def dtoh_async(self,**inputs) -> None:
        for output in self.outputs.values():
            cuda.memcpy_dtoh_async(
                output.host,
                output.device,
                self.stream
            )

    def htod_async(self) -> None:
        for _input in self.inputs.values():
            cuda.memcpy_htod_async(
                _input.device,
                _input.host,
                self.stream
            )

    async def async_exec(self,async_cuda_callable,*args,**kwargs) -> None:
        async_cuda_callable(*args,**kwargs)
        self._exec_event.record(self.stream)
        await self.sync_cuda_event_or_stream(self._exec_event)

    async def push(self,**inputs) -> None:
        async with self._input_guard:
            for tensor_name,_input in inputs.items():
                buffer = self.inputs[tensor_name]
                flat_input = _input.ravel()
                np.copyto(buffer.host[:flat_input.size],flat_input)
            self.htod_async()
            self._htod_event.record(self.stream)
            await self.sync_cuda_event_or_stream(self._htod_event)

    async def pull(self,**output_shapes) -> np.ndarray:
        async with self._output_guard:
            self.dtoh_async()
            self._dtoh_event.record(self.stream)
            await self.sync_cuda_event_or_stream(self._dtoh_event)
            outputs = {}
            for tensor_name,host_device_output in self.outputs.items():
                output = np.copy(host_device_output.host[:trt.volume(output_shapes[tensor_name])])  # This is a NumPy ndarray
                output = output.reshape(output_shapes[tensor_name])  # Reshape to original tensor shape
                outputs[tensor_name] = output
        return outputs