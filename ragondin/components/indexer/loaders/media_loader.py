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

MEDIA_FORMATS = [".wav", ".mp3", ".mp4", ".ogg", ".flv", ".wma", ".aac"]


class AudioTranscriber(metaclass=SingletonMeta):
    def __init__(self, device="cpu", compute_type="float32", model_name="base"):
        self.model = whisper.load_model(name=model_name, device=device)


class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs):
        super().__init__(page_sep, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = kwargs.get("config").loader["audio_model"]
        self.transcriber = AudioTranscriber(device=device, model_name=model)

    async def aload_document(
        self, file_path, metadata: dict = None, save_markdown=False
    ):
        path = Path(file_path)
        if path.suffix not in MEDIA_FORMATS:
            logger.warning(
                f"This audio/video file ({path.suffix}) is not supported."
                f"The format should be {MEDIA_FORMATS}"
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
        if save_markdown:
            self.save_document(Document(page_content=content), str(file_path))
        return doc
