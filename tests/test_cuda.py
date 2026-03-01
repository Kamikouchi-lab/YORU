"""Test to verify PyTorch works correctly in CUDA environment"""

import torch


def test_cuda_available():
    """Check if CUDA is available"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

    assert torch.cuda.is_available(), "CUDA is not available"


def test_cuda_tensor_operations():
    """Check if tensor operations work correctly on CUDA"""
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    device = torch.device("cuda")

    # Transfer tensors to GPU
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    c = torch.mm(a, b)

    # Transfer result back to CPU
    c_cpu = c.cpu()

    print("Tensor operation on GPU successful")
    print(f"  Result shape: {c_cpu.shape}")
    print(f"  Result dtype: {c_cpu.dtype}")

    assert c_cpu.shape == (1000, 1000)


def test_cuda_neural_network():
    """Check if a simple neural network works on CUDA"""
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    device = torch.device("cuda")

    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)

    # Dummy data
    x = torch.randn(32, 784, device=device)

    # Forward pass
    output = model(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    print("Neural network on GPU successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Loss: {loss.item():.4f}")

    assert output.shape == (32, 10)


def test_cuda_memory():
    """Check CUDA memory usage"""
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    torch.cuda.empty_cache()

    allocated_before = torch.cuda.memory_allocated()

    # Allocate memory
    tensor = torch.randn(10000, 10000, device="cuda")

    allocated_after = torch.cuda.memory_allocated()

    print("CUDA memory test:")
    print(f"  Allocated before: {allocated_before / 1024**2:.2f} MB")
    print(f"  Allocated after: {allocated_after / 1024**2:.2f} MB")
    print(f"  Tensor size: {(allocated_after - allocated_before) / 1024**2:.2f} MB")

    del tensor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch CUDA Test")
    print("=" * 50)

    test_cuda_available()
    print()
    test_cuda_tensor_operations()
    print()
    test_cuda_neural_network()
    print()
    test_cuda_memory()

    print()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
