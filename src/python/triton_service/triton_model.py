import subprocess
import time
from pathlib import Path
from os import environ as env
from functools import cached_property
from tritonclient.utils import InferenceServerException
from aiohttp.client_exceptions import ClientConnectorError
import tritonclient.grpc.aio as tritongrpcclient
from transformers import AutoTokenizer,AutoConfig
from typing import List,Dict,Optional
import numpy as np
import asyncio
import logging

def numpy_to_triton_dtype(np_dtype):
    # Map numpy dtypes to Triton format
    dtype_map = {
        np.int8: "INT8",
        np.int16: "INT16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.uint16: "UINT16",
        np.uint32: "UINT32",
        np.uint64: "UINT64",
        np.float16: "FP16",
        np.float32: "FP32",
        np.float64: "FP64",
        bool: "BOOL"
    }
    return dtype_map[np_dtype.type]


class TritonServer:
    def __init__(
        self,
        model_repository: str,
        http_port: int,
        grpc_port: int,
        metrics_port: int,
        gpu_id: int,
        log_dir: str

    ) -> None:
        cmd = [
            "tritonserver",
            f"--model-repository={model_repository}",
            f"--http-port={http_port}",
            f"--grpc-port={grpc_port}",
            f"--metrics-port={metrics_port}"
        ]
        triton_env = env.copy()
        triton_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.log_file = open(Path(log_dir)/"triton.log","w+")
        self.process = subprocess.Popen(
            cmd,
            env=triton_env,
            stdout=self.log_file,
            stderr=self.log_file
        )
        self.monitor_task = asyncio.get_event_loop().create_task(self.monitor())
    
    async def monitor(self,poll_interval_sec=5):
        while True:
            await asyncio.sleep(poll_interval_sec)
            if self.process.poll() not in (None,0):
                raise Exception("Triton server process has terminated with nonzero exit code")

    def shutdown(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log_file.close()
        if self.monitor_task:
            self.monitor_task.cancel()

class TritonServerWithAioClient:
    _client=None
    _output_names=None

    def __init__(
        self,
        model_repository: str,
        model_name: str,
        model_version: str,
        log_dir: str,
        http_port: int = 8000,
        grpc_port: int = 8001,
        metrics_port: int = 8002,
        gpu_id: int = 0
    ):
        self._model_ready=False
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_repository)/model_name/"tokenizer")
        self.hf_config = AutoConfig.from_pretrained(Path(model_repository)/model_name/"tokenizer")
        self.max_seq_len = self.hf_config.max_position_embeddings
        self.model_repository = Path(model_repository)
        self.model_name = model_name
        self.model_version = model_version
        self.http_port = http_port
        self.grpc_port = grpc_port 
        self.metrics_port = metrics_port
        self.log_dir = log_dir
        self.gpu_id = gpu_id

    @cached_property
    def logger(self):
        return logging.getLogger("app")

    async def predict(
        self, 
        texts: List[str]
    ) -> Dict[str, np.ndarray]:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.max_seq_len,
            return_tensors="np"
        )
        self.logger.debug(f"Inputs: {inputs}")
        infer_inputs = []            
        for name,data in inputs.items():
            infer_input = tritongrpcclient.InferInput(
                name,
                data.shape,
                numpy_to_triton_dtype(data.dtype)
            )
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input) 
        
        infer_outputs = [
            tritongrpcclient.InferRequestedOutput(name)
            for name in self.output_names
        ]
        self.logger.debug(f"Inputs: {inputs}\nInferInputs: {infer_inputs}\nInferOutputs: {infer_outputs}")
        response = await self.client.infer(
            model_name=self.model_name,
            inputs=infer_inputs,
            outputs=infer_outputs
        )
        self.logger.debug(f"Response: {response}")
        return {
            name: response.as_numpy(name)
            for name in self.output_names
        }

    @cached_property
    def event_loop(self):
        return asyncio.get_event_loop()

    @property
    def client(self,timeout=5):
        start_time = time.time()
        if self._client is None:
            while time.time() - start_time < timeout:
                if self._client is not None:
                    return self._client
                time.sleep(1)
        else: return self._client
        raise Exception("""
Triton client failed to initialize within timeout. 
The most likely cause is failure to await either:
  - start()
  - initialize_client()
Before using client property.
"""
        )

    @property
    def output_names(self,timeout=5):
        if self._output_names is None:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._output_names is not None:
                    return self._output_names
                time.sleep(1)
        else: return self._output_names
        raise Exception("""
Triton output_names failed to initialize within timeout. 
The most likely cause is failure to await either:
  - start()
  - load_model()
Before using output_names property.
"""
        )

    async def initialize_output_names(self):
        metadata = await self.client.get_model_metadata(self.model_name,self.model_version)
        self._output_names =  tuple(output.name for output in metadata.outputs)
        print(f"Output names: {self._output_names}")

    async def initialize_client(self):
        if self._client is None:
            self._client = tritongrpcclient.InferenceServerClient(
                url=f"0.0.0.0:{self.grpc_port}"
            )
            await self._client.__aenter__()
            

    async def load_model(self,timeout: int = 30) -> bool:
        model_ready = await self.client.is_model_ready(self.model_name,self.model_version)
        if not model_ready:
            await self.client.load_model(self.model_name)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    model_ready = await self.client.is_model_ready(self.model_name,self.model_version)
                    if model_ready:
                        self.logger.info("Model is ready")                        
                        return True
                except InferenceServerException:
                    await asyncio.sleep(1)
                    continue
            self.logger.error("Model failed to load within timeout")
            return False
        else:
            self.logger.info("Model is ready")
            return True

    async def start(self, timeout: int = 60) -> bool:
        self.logger.info("Starting Triton server subprocess...")
        self.server = TritonServer(
            self.model_repository,
            self.http_port,
            self.grpc_port,
            self.metrics_port,
            self.gpu_id,
            self.log_dir
        )
        start_time = time.time()
        await self.initialize_client()
        while time.time() - start_time < timeout:
            try:
                server_ready = await self.client.is_server_ready()
                if server_ready:                    
                    self._model_ready = await self.load_model()
                    if self._model_ready:
                        await self.initialize_output_names()
                        break
                    self.logger.info("Server is ready")
            except (InferenceServerException,ClientConnectorError):
                await asyncio.sleep(1)
                continue
        if not self._model_ready:
            self.logger.error("Server failed to start within timeout")
        
    @property
    def ready(self):
        return self._model_ready

    async def stop(self):
        if self.server:
            self.server.shutdown()
            await self.client.__aexit__(None, None, None)