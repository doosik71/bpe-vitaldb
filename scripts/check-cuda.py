import torch
import sys

def check_cuda():
    print("--- CUDA Availability Check ---")
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**2:.0f} MB")
        
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
    else:
        print("\nWARNING: CUDA is not available to PyTorch.")
        print("Training deep learning models on CPU will be significantly slower.")
        print("Please ensure you have a CUDA-compatible GPU and the correct drivers installed.")
        
        # Check if CUDA is even compiled into this torch version
        from torch.utils.cpp_extension import CUDA_HOME
        print(f"CUDA_HOME: {CUDA_HOME}")

if __name__ == "__main__":
    check_cuda()
