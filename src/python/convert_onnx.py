from convert_model import convert_hf_onnx_model
import onnx
from onnx import helper, numpy_helper, TensorProto


max_batch_size=64
optimal_batch_size=8
seq_len_optimization_factor=4


ADD_POOLER=True

def add_mean_pooling_and_norm_layer(onnx_model_path: str, output_model_path: str):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # Find the last_hidden_state output tensor
    last_hidden_state = None
    for output in graph.output:
        if output.name == "last_hidden_state":
            last_hidden_state = output
            break
    if last_hidden_state is None:
        raise ValueError("last_hidden_state output not found in the model")

    # Define the mean pooling layer
    mean_pooling_node = helper.make_node(
        'ReduceMean',
        inputs=[last_hidden_state.name],
        outputs=['mean_pooled'],
        axes=[1],  # Pooling over the sequence length dimension
        keepdims=False
    )

    # Define the normalization layer
    norm_node = helper.make_node(
        'LpNormalization',
        inputs=['mean_pooled'],
        outputs=['sentence_embeddings'],
        axis=1  # Normalize over the hidden dimension
    )

    # Add the new nodes to the graph
    graph.node.extend([mean_pooling_node, norm_node])

    # Add the new output tensor
    sentence_embeddings_output = helper.make_tensor_value_info(
        'sentence_embeddings',
        TensorProto.FLOAT,
        [None, last_hidden_state.type.tensor_type.shape.dim[2].dim_value]  # [batch_size, hidden_dim]
    )
    graph.output.append(sentence_embeddings_output)

    # Save the modified model
    onnx.save(model, output_model_path)

    
model_name = "model"
model_path_or_repo_id = "sentence-transformers/all-MiniLM-L12-v2"
if ADD_POOLER:
    new_model_name = "sentence_transformer_model"
    add_mean_pooling_and_norm_layer(
        f"{model_path_or_repo_id}/onnx/{model_name}.onnx", 
        f"{model_path_or_repo_id}/onnx/{new_model_name}.onnx"
    )
    model_name = new_model_name


onnx_model = f"{model_name}.onnx"
trt_model = f"{model_name}.trt"

convert_hf_onnx_model(
    model_path_or_repo_id,
    onnx_model,
    trt_model,
    max_batch_size,
    optimal_batch_size,
    seq_len_optimization_factor
)