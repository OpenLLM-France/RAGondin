import os
from pathlib import Path
import whisper
from langchain_core.documents.base import Document
from loguru import logger
from pydub import AudioSegment
from .base import BaseLoader

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 4
        self.page_sep = page_sep
        self.formats = [".wav", ".mp3", ".mp4"]

        self.model = whisper.load_model("base")

    async def aload_document(self, file_path, metadata: dict = None, save_md=False):
        path = Path(file_path)
        if path.suffix not in self.formats: 
            logger.warning(
                f"This audio/video file ({path.suffix}) is not supported."
                f"The format should be {self.formats}"
            )
            return None

        # load it in wave format
        if path.suffix == ".wav":
            audio_path_wav = path
        else:
            sound = AudioSegment.from_file(file=path, format=path.suffix[1:])

            audio_path_wav = path.with_suffix(".wav")  # Export to wav format
            sound.export(
                audio_path_wav,
                format="wav",
            )

        logger.info(f"SOUND: {file_path}")
        result = self.model.transcribe(str(audio_path_wav))

        if path.suffix != ".wav":
            os.remove(audio_path_wav)

        content = result["text"]
        doc = Document(page_content=content, metadata=metadata)
        if save_md:
            self.save_document(Document(page_content=content), str(file_path))
        return doc
