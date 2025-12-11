import torch
from tron.ai.tron_model import CnnTronModel

model = CnnTronModel(5, 5)


torch.save(
    model.state_dict(),
    r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\models\20250810_5x5_random_init.pth",
)
