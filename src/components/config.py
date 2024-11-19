import os
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra import initialize, compose

def load_config(config_path="../../.hydra_config")-> OmegaConf:
    
    load_dotenv()

    with initialize(config_path=config_path, job_name="config_loader"):
        config = compose(config_name="config")
        dir_path = Path(__file__).parent
        config.dir_path = dir_path
        config.prompts_dir = dir_path / 'prompts'
        config.llm.api_key = os.environ["API_KEY"]
        return config

# Example usage
if __name__ == "__main__":
    config = load_config()
    print(config)