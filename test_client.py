from typing import Optional

import torch
import omegaconf
from loguru import logger
import os
import matplotlib.pyplot as plt
import psutil  # For memory usage tracking

from training.trainer import Trainer
from models import get_model
from datasets.guidedepth_dataset import GuideDepthDataset
from training.metrics import plot_metrics

from PEERNet_fl.peernet.networks import ZMQ_Pair

# Initialize Model
model_name = "hybrid"
model_params = {"in_channels":3, "height":240, "width":320}
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_model(model_name, **model_params).to(device)

training_data_paths = ["B:\\andre\\Documents\\Fall 2024 Classes\\Senior_Design\\edgemapper-client\\captured_images"]
val_data_path = "B:\\andre\\Documents\\Fall 2024 Classes\\Senior_Design\\edgemapper-client\\captured_images"

trainer = Trainer(
    model,
    model_name,
    training_data_paths[0],
    device,
    4,
    1e-4,
    val_data_path
)
logger.info(f"Initialized trainer on {device}")

global_epoch = 0
local_epoch = 0
max_local_epochs = 8


global_metrics = {}
for idx, training_path in enumerate(training_data_paths):
    trainer.train(max_local_epochs)
    trainer.plot_val()
    local_metrics = trainer.validate()
    # global_metrics.update({idx, local_metrics})
    trainer.update_dataset(training_path)
trainer.plot_results()
# plot_metrics(global_metrics, "./results")

# Print memory usage information
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
logger.info(f"Memory Usage Summary:")
logger.info(f"RSS (Resident Set Size): {memory_info.rss / (1024 * 1024):.2f} MB")
logger.info(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 * 1024):.2f} MB")
if hasattr(memory_info, 'gpu_memory'):
    logger.info(f"GPU Memory: {memory_info.gpu_memory / (1024 * 1024):.2f} MB")

# Report GPU memory if using CUDA
if torch.cuda.is_available():
    logger.info(f"CUDA Memory Summary:")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i} - Allocated: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB")
        logger.info(f"GPU {i} - Cached: {torch.cuda.memory_reserved(i) / (1024 * 1024):.2f} MB")