import torch

gpu_id = torch.cuda.current_device()
print(torch.cuda.device(gpu_id))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(gpu_id))
