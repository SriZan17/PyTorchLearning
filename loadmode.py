import torch
import torch.nn as nn
from learning import LRModel2

model = LRModel2()
model.load_state_dict(torch.load("models/model0.pth"))
model.eval()
print(model.state_dict())
