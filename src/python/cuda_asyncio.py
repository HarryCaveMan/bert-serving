import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import asyncio

def allocate_buffers(context_wrapper):
    inputs,outputs,bindings = [],[],[]
    for binding_index in range(context_wrapper._trt_context.engine.num_bindings):
        binding_name = context_wrapper._trt_context.engine.get_binding_name(binding_index)
        shape = context_wrapper._trt_context.engine.get_binding_shape(binding_name)
        size = trt.volume(shape)
        dtype = trt.nptype(context_wrapper._trt_context.engine.get_binding_dtype(binding_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        mem_obj = HostDeviceMem(host_mem, device_mem, shape)
        if context_wrapper._trt_context.engine.binding_is_input(binding_name):
            inputs.append(mem_obj)
        else:
            outputs.append(mem_obj)
    return IOBufferSet(
        inputs,
        outputs,
        bindings,
        context_wrapper.stream,
        context_wrapper._io_thread_pool
    )

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
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self._taints = 0

    def taint(self) -> None:
        self._taints += 1

    def clean(self) -> None:
        self._taints -= 1

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
    async def sync_cuda_event(self,event) -> None:
        await self._loop.run_in_executor(
            self._thread_pool,
            self.stream.wait_for_event,
            event
        )

    def dtoh_async(self) -> None:
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
        htod_event = cuda.Event()
        async with self._input_guard:
            np.copyto(self.inputs[input_index].host, input_data.ravel())
            self.htod_async(input_index)
            self.stream.record(htpd_event)
            await self.sync_cuda_event(htod_event)

    async def pull(self,event) -> np.ndarray:
        async with self._output_guard:
            self.dtoh_async()
            event.record(self.stream)
            await self.sync_cuda_event(event)
            outputs = []
            for host_device_output in self.outputs:
                output = np.copy(host_device_output.host)  # This is a NumPy ndarray
                output = output.reshape(host_device_output.shape)  # Reshape to original tensor shape
                outputs.append(output)            
            return outputs