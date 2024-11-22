import os
from dotenv import load_dotenv, find_dotenv
from omegaconf import OmegaConf
from hydra import initialize, compose

def load_config(config_path="../../.hydra_config", overrides=None)-> OmegaConf:
    load_dotenv()
    print(overrides)
        
    with initialize(config_path=config_path, job_name="config_loader"):
        config = compose(config_name="config", overrides=overrides)
        return config

# # Example usage
# if __name__ == "__main__":
#     config = load_config()
#     print(config)