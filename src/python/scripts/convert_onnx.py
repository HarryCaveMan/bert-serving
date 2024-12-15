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


    # Cast mask to float
    cast_mask = helper.make_node(
        'Cast',
        inputs=['attention_mask'],
        outputs=['float_mask'],
        to=1  # FLOAT
    )

    # Expand attention mask to match hidden state dimensions
    expand_mask = helper.make_node(
        'Unsqueeze',
        inputs=['float_mask'],
        outputs=['expanded_mask'],
        axes=[2]
    )

    # Multiply hidden states with mask
    masked_hidden = helper.make_node(
        'Mul',
        inputs=[last_hidden_state.name, 'expanded_mask'],
        outputs=['masked_hidden_states']
    )

    # Sum over sequence length (numerator)
    sum_embeddings = helper.make_node(
        'ReduceSum',
        inputs=['masked_hidden_states'],
        outputs=['sum_embeddings'],
        axes=[1],  # Sum over sequence length
        keepdims=False
    )

    # Sum attention mask (denominator)
    token_count = helper.make_node(
        'ReduceSum',
        inputs=['float_mask'],
        outputs=['token_count'],
        axes=[1]  # Sum over sequence length
    )

    # Compute mean by division
    mean_pooling = helper.make_node(
        'Div',
        inputs=['sum_embeddings', 'token_count'],
        outputs=['mean_pooled']
    )

    # Define the normalization layer
    norm = helper.make_node(
        'LpNormalization',
        inputs=['mean_pooled'],
        outputs=['sentence_embeddings'],
        axis=1  # Normalize over the hidden dimension
    )

    # Add the new nodes to the graph
    graph.node.extend(
        [
            cast_mask,
            expand_mask,
            masked_hidden,
            sum_embeddings,
            token_count,
            mean_pooling,
            norm
        ]
    )

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
    new_model_name = "model_10_6.trt"
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