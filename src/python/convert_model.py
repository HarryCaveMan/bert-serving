import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from huggingface_hub import snapshot_download
import os


def modify_onnx_remove_mask(input_path: str, output_path: str):
    model = onnx.load(input_path)
    
    # Remove attention_mask input
    inputs = [input for input in model.graph.input if input.name != 'attention_mask']
    model.graph.ClearField('input')
    model.graph.input.extend(inputs)
    
    # Update pooling to use mean across sequence dim
    nodes = []
    
    # Mean pooling without mask
    nodes.append(helper.make_node(
        'ReduceMean',
        inputs=['last_hidden_state'],
        outputs=['pooled'],
        axes=[1],  # sequence dimension
        name='mean_pooling'
    ))
    
    # Normalize embeddings
    nodes.append(helper.make_node(
        'LpNormalization',
        inputs=['pooled'],
        outputs=['normalized_embeddings'],
        axis=1,
        p=2
    ))
    
    # Update graph
    model.graph.node.extend(nodes)
    model.graph.output[0].name = 'normalized_embeddings'
    
    onnx.save(model, output_path)

def build_engine(onnx_path: str, engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1024 * 1024 * 1024
    
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_ids",
        min=(1, 32),
        opt=(8, 128),
        max=(128, 512)
    )
    
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    return engine