from typing import Optional

import torch
import omegaconf
from loguru import logger

from training.trainer import Trainer
from models import get_model

from PEERNet_fl.peernet.networks import ZMQ_Pair


class FLClient:
    """
    A client for federated learning, responsible for local training and model updates.

    Args:
        server_url (str): The URL of the federated server.
        model_name (str): Name of the model to train.
        device (torch.device): Computation device (CPU or GPU).
        data_path (str): Path to the local dataset.
    """

    def __init__(self,
                 device_name: str,
                 model_name: str,
                 training_data_path: str,
                 val_data_path: Optional[str],
                 batch_size: int = 8,
                 local_epochs: int = 15,
                 lr: float = 1e-4,
                 model_params=None,
                 loss_func=None):

        # Establish connection to server
        self.device_name = device_name
        self.net_config = omegaconf.OmegaConf.load("net_config.yaml")
        self.network = ZMQ_Pair(device_name=device_name,
                                start_port=10000,
                                **self.net_config)
        logger.info("Connected to server")

        # Initialize Model
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(model_name, model_params).to(self.device)

        self.trainer = Trainer(
            self.model,
            model_name,
            training_data_path,
            self.device,
            batch_size,
            lr,
            val_data_path,
            loss_func,
        )
        logger.info(f"Initialized trainer on {self.device}")

        self.global_epoch = 0
        self.local_epoch = 0
        self.max_local_epochs = local_epochs

    def _download_model(self):
        # Pytorch load state dict for the model
        self.global_epoch, server_gradients = self.network.recv("server")
        server_gradients = {
            k: v.to(self.device)
            for k, v in server_gradients.items()
        }
        self.model.load_state(server_gradients)
        logger.info("Succesfully received model from server")

    def _upload_gradients(self):
        # Send model weights to server
        gradients = self.model.get_state()
        cpu_state_dict = {k: v.cpu() for k, v in gradients.items()}
        self.network.send("server", cpu_state_dict)
        logger.info("Successfully sent gradients to server.")

    def run(self):
        while True:
            logger.info(f"Staring global round: {self.global_epoch}")
            self._download_model()

            self.trainer.train(self.max_local_epochs)
            self.trainer.validate()
            self.trainer.plot_results()
            self._upload_gradients()
            # TODO: Update dataset with new images/delete old ones. I'm assuming there will be some new path to new images we can know.
            self.trainer.update_dataset("path to new images")
