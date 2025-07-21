from pathlib import Path
import torch



from tron.ai.model_architectures_old import EvaluationNetConv3OneStride

device = torch.device("cpu")

state_dict = torch.load(
    "C:/Users/kylan/Documents/code/repos/Tron/python/tasks/2024_12_09_eval/runs/20241211-171205_oldnet_self_train_continuation_v5/checkpoints/oldnet_self_train_continuation_v5_7.pth"
)
torch_model = EvaluationNetConv3OneStride(grid_dim=10)
torch_model.load_state_dict(state_dict)
torch_model = torch_model.to(device)

torch_input = torch.randn(1, 3, 10, 10).to(device)


# onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
# onnx_program.save("./tron_model_v2.onnx")


torch.onnx.export(
    torch_model,
    torch_input,
    "tron_model_v2.onnx",
    opset_version=18,  # or a compatible opset version
    input_names=["input"],
    output_names=["output"]
)