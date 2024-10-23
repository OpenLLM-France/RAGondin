import os
import configparser
from pathlib import Path

class Config:
    """This class encapsulates the configurations for the application, 
    including settings for paths, directories, and LLM-specific parameters.
    """
    def __init__(self, cfg_file) -> None:
        self.api_key: str = os.getenv('API_KEY')
        self.dir_path = Path(__file__).parent
        self.prompts_dir = self.dir_path / 'prompts'

        if not self.api_key:
            print("Error: API_KEY not set")
        
        parser = configparser.ConfigParser()
        parser.read_file(open(cfg_file))

        for key, value in parser.items():
            params = dict(value.items())
            setattr(self, key, params)
