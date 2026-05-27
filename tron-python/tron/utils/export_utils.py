from pathlib import Path

import numpy as np
import torch

from tron.ai.nnue import QuantizedNnueTronModel


def export_quantized_nnue(
    model: QuantizedNnueTronModel,
    path: str | Path,
    dtype=np.int64,
) -> None:
    assert isinstance(model, QuantizedNnueTronModel)

    data = {
        "scale": np.array([model.scale], dtype=dtype),
        "padding_idx": np.array([model.raw_model.padding_idx], dtype=dtype),
        "embed_weights": model.embed_weights.cpu().numpy().astype(dtype),
        "fc_value_weights": model.fc_value_weights.cpu().numpy().astype(dtype),
        "fc_value_bias": model.fc_value_bias.cpu().numpy().astype(dtype),
    }

    for i, weight in enumerate(model.fc_layer_weights):
        data[f"fc_layer_{i}_weights"] = weight.cpu().numpy().astype(dtype)

    for i, bias in enumerate(model.fc_layer_biases):
        data[f"fc_layer_{i}_bias"] = bias.cpu().numpy().astype(dtype)

    data["num_fc_layers"] = np.array([len(model.fc_layer_weights)], dtype=dtype)

    np.savez(path, **data)


# def export_onnx():

#     device = torch.device("cpu")

#     state_dict = torch.load(
#         "C:/Users/kylan/Documents/code/repos/Tron/python/tasks/2024_12_09_eval/runs/20241211-171205_oldnet_self_train_continuation_v5/checkpoints/oldnet_self_train_continuation_v5_7.pth"
#     )
#     torch_model = EvaluationNetConv3OneStride(grid_dim=10)
#     torch_model.load_state_dict(state_dict)
#     torch_model = torch_model.to(device)

#     torch_input = torch.randn(1, 3, 10, 10).to(device)


#     # onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
#     # onnx_program.save("./tron_model_v2.onnx")


#     torch.onnx.export(
#         torch_model,
#         torch_input,
#         "tron_model_v2.onnx",
#         opset_version=18,  # or a compatible opset version
#         input_names=["input"],
#         output_names=["output"]
#     )
