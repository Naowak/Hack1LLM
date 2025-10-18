import torch

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"{num_devices} CUDA device(s) available:\n")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB\n")
else:
    print("No CUDA-compatible GPU detected.")