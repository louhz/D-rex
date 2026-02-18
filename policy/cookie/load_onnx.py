import onnx
import torch
import torch.nn as nn
import pprint

# Load ONNX model
def load_onnx_model(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model

# Convert model to Python data structure
def model_to_dict(model):
    model_dict = {
        'graph': {
            'name': model.graph.name,
            'inputs': [],
            'outputs': [],
            'nodes': []
        }
    }

    for input_tensor in model.graph.input:
        model_dict['graph']['inputs'].append({
            'name': input_tensor.name,
            'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
            'type': input_tensor.type.tensor_type.elem_type
        })

    for output_tensor in model.graph.output:
        model_dict['graph']['outputs'].append({
            'name': output_tensor.name,
            'shape': [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim],
            'type': output_tensor.type.tensor_type.elem_type
        })

    for node in model.graph.node:
        node_info = {
            'name': node.name,
            'op_type': node.op_type,
            'inputs': list(node.input),
            'outputs': list(node.output)
        }
        model_dict['graph']['nodes'].append(node_info)

    return model_dict

# Automatically convert ONNX nodes to PyTorch model (basic example for Linear and ReLU layers)
def onnx_to_pytorch(model_dict):
    layers = []
    for node in model_dict['graph']['nodes']:
        if node['op_type'] == 'Gemm' or node['op_type'] == 'MatMul':
            # Assuming weights/biases dimensions from inputs (you may need more parsing for production)
            in_features = model_dict['graph']['inputs'][0]['shape'][1]
            out_features = model_dict['graph']['outputs'][0]['shape'][1]
            layers.append(nn.Linear(in_features, out_features))
        elif node['op_type'] == 'Relu':
            layers.append(nn.ReLU())
        # Add more operations as needed

    return nn.Sequential(*layers)

if __name__ == "__main__":
    model_path = "cookie/mnist_3layer_batch.onnx"

    model = load_onnx_model(model_path)
    model_dict = model_to_dict(model)

    # Pretty print the Python data structure
    pprint.pprint(model_dict)

    # Automatically create PyTorch model from ONNX structure
    pytorch_model = onnx_to_pytorch(model_dict)

    # Save the PyTorch model explicitly
    torch.save(pytorch_model.state_dict(), "converted_model.pth")