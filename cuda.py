import torch

# 检测CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA is available!")

    # 获取GPU数量并列出每个GPU的名称
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

