from pathlib import Path
import torch

model_path = Path("C://Users//kylan//Documents//code//repos//KOTron//python//tasks//2024_12_refactor//0728_random_train_one_stride_19.pt")

# model = torch.load(model_path).to("cpu")


model = torch.load(model_path)

torch.save(model.state_dict(), 'model_state.pth')

