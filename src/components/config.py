import os
from dataclasses import field
import configparser
from pathlib import Path

class Config:
    """This class encapsulates the configurations for the application, 
    including settings for paths, directories, and LLM-specific parameters.
    """
    def __init__(self, cfg_file) -> None:
        # TODO: place chunker args in config.ini 
        self.chunker_name: str = "semantic_splitter" # "semantic_splitter"
        self.chunk_size: int = 1000
        self.chunk_overlap: int = 100 # TODO: Better express it with a percentage
        # self.chunker_args: dict = field(default_factory=dict) # additional attributes specific to chunker

        self.api_key: str = os.getenv('API_KEY')
        self.dir_path = Path(__file__).parent

        if not self.api_key:
            print("Error: API_KEY not set")
        
        parser = configparser.ConfigParser()
        parser.read_file(open(cfg_file))

        for key, value in parser.items():
            setattr(self, key, dict(value.items()))