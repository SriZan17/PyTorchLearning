import torch

seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(seed)
x = torch.rand(2, 3).to(device)
torch.cuda.manual_seed(seed)
y = torch.rand(2, 3).to(device).transpose(0, 1)
x = x @ y
print(x)
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device("cuda")
# else:
#     args.device = torch.device("cpu")
