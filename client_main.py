from fl_client import FLClient
from omegaconf import OmegaConf
from loguru import logger

config = OmegaConf.from_cli()

device = config.get("device")

logger.debug(f"Using device name {device}")

client = FLClient(device_name=device)
client.run()