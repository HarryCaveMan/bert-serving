import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from pathlib import Path

def build_engine(
    onnx_path:str,
    engine_path:str,
    hidden_dim:int,
    max_seq_len:int,
    max_batch_size:int,
    optimal_batch_size:int,
    seq_len_optimization_factor:int,
    device_mem:float=.9
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        onnx_model = onnx.load(onnx_path)
        print(f"ONNX inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"ONNX outputs: {[o.name for o in onnx_model.graph.output]}")        
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX")            
    for input_idx in range(network.num_inputs):
        _input = network.get_input(input_idx)
        print(f"""
        TRT Input: {_input.name}
        Shape: {_input.shape}
        """)
    for output_idx in range(network.num_outputs):
        output = network.get_output(output_idx)
        print(f"""
        TRT Output: {output.name}
        Shape: {output.shape}
        """)
        if output.name == "last_hidden_state":
            dims = output.shape
            if len(dims) != 3 or dims[2] != hidden_dim:
                print(
                    f"Invalid last_hidden_state shape after parsing: {dims}. "
                    f"Expected (..., {hidden_dim})"
                )
    _,total_device_mem = cuda.mem_get_info()
    config = builder.create_builder_config()
    workspace_size = int(total_device_mem*device_mem)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)    
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_ids",
        min=(1, 1),
        opt=(optimal_batch_size, max(1,int(max_seq_len/seq_len_optimization_factor))),
        max=(max_batch_size, max_seq_len)
    )
    profile.set_shape(
        "attention_mask",  # Add shape for attention mask
        min=(1, 1),
        opt=(optimal_batch_size, max(1,int(max_seq_len/seq_len_optimization_factor))),
        max=(max_batch_size, max_seq_len)
    )
    profile.set_shape(
        "token_type_ids",  # Add shape for attention mask
        min=(1, 1),
        opt=(optimal_batch_size, max(1,int(max_seq_len/seq_len_optimization_factor))),
        max=(max_batch_size, max_seq_len)
    )
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    return engine


def convert_hf_onnx_model(model_path_or_repo_id:str,onnx_model:str,trt_model:str,max_batch_size:int,optimal_batch_size:int,seq_len_optimization_factor:int):
    local_dir:str = model_path_or_repo_id.split("/")[0]
    Path(local_dir).absolute().mkdir(mode=0o755,parents=True,exist_ok=True)
    onnx_dir:str = f"{model_path_or_repo_id}/onnx"
    trt_dir:str = f"{model_path_or_repo_id}/trt"
    Path(trt_dir).absolute().mkdir(mode=0o755,parents=True,exist_ok=True)
    if not Path(onnx_dir,"model.onnx").exists():
        snapshot_download(model_path_or_repo_id,local_dir=model_path_or_repo_id)
    model_config = AutoConfig.from_pretrained(model_path_or_repo_id,trust_remote_code=True)
    max_seq_len = model_config.max_position_embeddings
    hidden_dim = model_config.hidden_size
    trt_cuda_engine = build_engine(
        f"{onnx_dir}/{onnx_model}",
        f"{trt_dir}/{trt_model}",
        hidden_dim,
        max_seq_len,
        max_batch_size,
        optimal_batch_size,
        seq_len_optimization_factor
    )