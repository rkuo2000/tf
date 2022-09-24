import torch

print("GPU CUDA available  ? %s" % torch.cuda.is_available())
print("CUDA device count   : %d" % torch.cuda.device_count())
print("CUDA current device : %d" % torch.cuda.current_device())
n = torch.cuda.current_device()
print("CUDA device name    : %s" % torch.cuda.get_device_name(n))

