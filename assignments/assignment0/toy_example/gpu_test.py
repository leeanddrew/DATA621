# gpu_test.py

import torch

def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available.")
        # Perform a simple tensor operation on the GPU
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        z = x + y
        print(f"Result of tensor addition on GPU: {z}")
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    main()
