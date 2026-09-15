import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(5000, 5000, device="cuda")
    y = torch.randn(5000, 5000, device="cuda")
    z = x @ y

    print("Computation device:", z.device)
    print("GPU computation successful!")