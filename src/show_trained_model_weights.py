import torch
from models.mlp_fashion_model import FashionMNIST_MLP

model = FashionMNIST_MLP()
model.load_state_dict(torch.load('../results/mlp_fashion_trainierte_model.pth', map_location='cpu'))

print("\nGewichten:\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
    print(param)
    print("-" * 50)
