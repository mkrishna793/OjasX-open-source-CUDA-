"""
OjasX MNIST Training Example (v2)
===================================
Trains a simple neural network on MNIST using TorchCL's OpenCL backend
with a 100% GPU-resident autograd pipeline (no CPU fallbacks).
"""

import time
import torch
import torch.nn as nn
import numpy as np

import torchcl
from torchcl.autograd import CrossEntropyFunction

print()
print("=" * 65)
print("  OjasX - GPU-Resident MNIST Neural Network Training on OpenCL")
print("=" * 65)
print()

info = torchcl.get_device_info()
print(f"  Device: {info['name']}")
print(f"  Memory: {info['global_mem_size_mb']} MB | CUs: {info['max_compute_units']}")
print()


# ── Simple MLP for digit classification ──────────────────────────────

class SimpleMLP(nn.Module):
    """A 3-layer MLP that runs entirely on OpenCL."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# ── Generate synthetic MNIST-like data ───────────────────────────────

def generate_batch(batch_size: int = 64, num_classes: int = 10):
    """Generate a batch of synthetic MNIST-like data."""
    images = torch.zeros(batch_size, 784)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        label = i % num_classes
        labels[i] = label

        # Create a distinct pattern for each digit class
        start_row = (label * 2) % 20
        start_col = (label * 3) % 20

        img = torch.zeros(28, 28)
        img[start_row:start_row+8, start_col:start_col+8] = torch.randn(8, 8) * 0.5 + 1.0
        images[i] = img.flatten()

    return images, labels


def one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert labels to one-hot encoding."""
    batch_size = labels.shape[0]
    oh = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        oh[i, labels[i]] = 1.0
    return oh


# ── Training Loop ────────────────────────────────────────────────────

def train():
    # 1. Initialize model and move it to OpenCL
    model = SimpleMLP()
    model.to("opencl")

    lr = 0.1
    num_epochs = 40
    batch_size = 128

    print(f"  Training for {num_epochs} epochs, batch_size={batch_size}")
    print(f"  Architecture: 784 -> 256 -> 128 -> 10")
    print()
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'Accuracy':>8} | {'Time':>8}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8}")

    total_start = time.time()
    engine = torchcl.ops.engine.get_engine()
    pool = torchcl.runtime.memory.get_buffer_pool()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Generate training batch
        images, labels = generate_batch(batch_size)

        # Move tensors to OpenCL (OjasXTensor subclass wrappers)
        x = images.to("opencl")
        t = labels.float().to("opencl")

        # Zero gradients
        for p in model.parameters():
            p.grad = None

        # Forward pass (runs fully on GPU via LinearFunction/ReluFunction)
        logits = model(x)

        # Compute loss on GPU using CrossEntropyFunction (passing wrappers directly)
        loss = CrossEntropyFunction.apply(logits, t)

        # Backward pass (fully GPU-resident)
        loss.backward()

        # GPU-native SGD Optimizer Step
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p_buf = torchcl.api._get_buf(p)
                    grad_buf = torchcl.api._get_buf(p.grad)
                    n = int(np.prod(p.shape))
                    
                    # Compute scaled gradient on GPU: scaled_grad = lr * grad
                    scaled_grad_buf = pool.allocate(n * 4, np.dtype(np.float32), p.shape)
                    engine.run_elementwise_scalar("mul_scalar_f32", grad_buf, lr, scaled_grad_buf, n)
                    
                    # Update parameter in-place on GPU: p = p - scaled_grad
                    engine.run_elementwise_binary("sub_f32", p_buf, scaled_grad_buf, p_buf, n)
                    pool.free(scaled_grad_buf)
                    
                    p.grad = None

        # Fetch loss and predictions to CPU for logging
        loss_cpu = torchcl.to_cpu(loss).item()
        probs_cpu = torchcl.to_cpu(logits)
        predictions = probs_cpu.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        epoch_time = (time.time() - epoch_start) * 1000.0

        if epoch % 2 == 0 or epoch == num_epochs - 1:
            print(f"  {epoch+1:>5} | {loss_cpu:>8.4f} | {accuracy:>7.1%} | {epoch_time:>6.1f}ms")

    total_time = time.time() - total_start
    print()
    print(f"  Training complete in {total_time:.2f}s")
    print(f"  Final loss: {loss_cpu:.4f}")
    print(f"  Final accuracy: {accuracy:.1%}")

    # Verify model has converged/learned the pattern
    assert accuracy > 0.8, f"Accuracy {accuracy:.1%} is too low — model did not converge properly!"
    print("  [SUCCESS] Model successfully converged on OpenCL!")
    print()
    print("=" * 65)


if __name__ == "__main__":
    train()
