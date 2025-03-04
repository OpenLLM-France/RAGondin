from dotenv import load_dotenv
from omegaconf import OmegaConf
from pathlib import Path
from hydra import initialize, compose
from loguru import logger

def load_config(config_path="../../.hydra_config", overrides=None)-> OmegaConf:
    load_dotenv()
    logger.info(f"Config overrides: {overrides}")
        
        # TODO: I set the version base to 1.1 to silence the warning message, review how we want to handle versioning
    with initialize(config_path=config_path, job_name="config_loader", version_base='1.1'):
        config = compose(config_name="config", overrides=overrides)
        config.paths.root_dir = Path(config.paths.root_dir).absolute()
        return config