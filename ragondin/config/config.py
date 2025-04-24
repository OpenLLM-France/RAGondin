from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf


def load_config(config_path="../../.hydra_config", overrides=None) -> OmegaConf:
    load_dotenv()
    if overrides:
        logger.info(f"Config overrides: {overrides}")

        # TODO: I set the version base to 1.1 to silence the warning message, review how we want to handle versioning
    with initialize(
        config_path=config_path, job_name="config_loader", version_base="1.1"
    ):
        config = compose(config_name="config", overrides=overrides)
        config.paths.data_dir = Path(config.paths.data_dir).resolve()
        config.paths.volumes_dir = Path(config.paths.volumes_dir).resolve()
        return config
