# Ray Cluster - Simple and Pythonic
import ray
import torch
import torch.nn as nn
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# Initialize Ray cluster
ray.init(address="auto")  # Connects to existing Ray cluster


# Define your training function
@ray.remote(num_gpus=1)
def train_model(config):
    """Distributed training task - Ray handles scheduling automatically"""
    model = nn.Linear(config["input_size"], config["output_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    for epoch in range(config["epochs"]):
        # Your training code here
        loss = torch.randn(1)  # Placeholder
        optimizer.step()

        # Ray automatically aggregates metrics across workers
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model.state_dict()


# Launch distributed training with Ray
config = {"input_size": 128, "output_size": 10, "lr": 0.001, "epochs": 10}

# Submit 4 parallel training jobs (e.g., for hyperparameter search)
futures = [train_model.remote(config) for _ in range(4)]

# Collect results - Ray handles data transfer
results = ray.get(futures)

# ============================================
# Using Ray Train for more advanced distributed training
# ============================================


def train_func(config):
    """Training function for Ray Train (data parallel)"""
    model = nn.Linear(128, 10)
    model = train.torch.prepare_model(model)  # Wraps for distributed

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(10):
        # Ray Train handles data sharding automatically
        loss = torch.randn(1)
        optimizer.step()

        # Report metrics - Ray aggregates across workers
        train.report({"loss": loss.item(), "epoch": epoch})


# ============================================
# Ray Data Pipeline - efficient data loading
# ============================================

# Load and preprocess data in parallel
ds = ray.data.read_parquet("s3://my-bucket/data/*.parquet")
ds = ds.map_batches(lambda batch: preprocess(batch), batch_size=256)
ds = ds.random_shuffle()  # Distributed shuffle

# Use with Ray Train automatically
trainer = TorchTrainer(
    train_func,
    datasets={"train": ds},  # Ray handles data distribution
    scaling_config=ScalingConfig(
        num_workers=4, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}  # 4 GPUs
    ),
)

# Run distributed training
result = trainer.fit()
print(f"Training complete: {result.metrics}")
