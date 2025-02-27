from fl_client import FLClient
from omegaconf import OmegaConf
from loguru import logger

config = OmegaConf.from_cli()

device = config.get("device")
model_name = config.get("model_name")
training_data_path = config.get("training_data_path")
val_data_path = config.get("val_data_path")

logger.debug(f"Using device name {device}")

client = FLClient(device_name=device,
                  model_name=model_name,
                  training_data_path=training_data_path,
                  val_data_path=val_data_path)
client.run()
