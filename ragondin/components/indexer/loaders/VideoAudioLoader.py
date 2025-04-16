import os
from pathlib import Path
import whisper
from langchain_core.documents.base import Document
from loguru import logger
from pydub import AudioSegment
from .base import BaseLoader
from components.utils import SingletonMeta


import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AudioTranscriber(metaclass=SingletonMeta):
    def __init__(self, device="cpu", compute_type="float32", model_name="base"):
        # self.model = whisperx.load_model(
        #     model_name, device=device, language=language, compute_type=compute_type
        # )
        self.model = whisper.load_model(name=model_name, device=device)


class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 4
        self.page_sep = page_sep
        self.formats = [".wav", ".mp3", ".mp4"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriber = AudioTranscriber(device=device, model_name="base")

    @classmethod
    def destroy(cls):
        if AudioTranscriber in SingletonMeta._instances:
            del SingletonMeta._instances[AudioTranscriber]

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
        result = self.transcriber.model.transcribe(str(audio_path_wav))

        if path.suffix != ".wav":
            os.remove(audio_path_wav)

        content = result["text"]
        doc = Document(page_content=content, metadata=metadata)
        if save_md:
            self.save_document(Document(page_content=content), str(file_path))
        return doc
