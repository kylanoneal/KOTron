from pathlib import Path
import torch

model_path = Path("C:/Users/kylan/Documents/code/repos/KOTron/python/model_checkpoints/0728_random_train_one_stride/0728_random_train_one_stride_19.pt")

# model = torch.load(model_path).to("cpu")

# torch_input = torch.randn(1, 3, 10, 10).to("cpu")
# onnx_program = torch.onnx.dynamo_export(model, torch_input)
# onnx_program.save("./tron_model.onnx")


# Assuming your trained model is `model`
# model.eval()
# scripted_model = torch.jit.script(model)  # or torch.jit.trace(model, example_inputs)
# scripted_model.save("tron_torchscript_model.pt")


# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('./model_scripted.pt') # Save

# print('bp')

import torch
from torch.onnx import ExportOptions

# model = torch.load(model_path).to("cpu")
# sample_input = torch.randn(1, 3, 10, 10).to("cpu")

# # Enable dynamic shapes in the export options
# options = ExportOptions(dynamic_shapes=True)

# onnx_program = torch.onnx.dynamo_export(
#     model,
#     sample_input,
#     export_options=options
# )

# onnx_program.save("tron_model_dynamic.onnx")


model = torch.load(model_path).to("cpu")
# Set the model to evaluation mode
model.eval()

# Create a dummy input with a batch size of 1
dummy_input = torch.randn(1, 3, 10, 10)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "tron_model_dynamic.onnx", 
                  input_names=["input"], 
                  output_names=["output"], 
                  dynamic_axes={"input": {0: "batch_size"}, 
                                "output": {0: "batch_size"}})