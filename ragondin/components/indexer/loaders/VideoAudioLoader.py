import gc
import os
import torch
import whisperx
from pathlib import Path
from loguru import logger
from pydub import AudioSegment

from langchain_core.documents.base import Document

from .base import BaseLoader
from .AudioTranscriber import AudioTranscriber

class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', batch_size=4, config=None):    
        self.batch_size = batch_size
        self.page_sep = page_sep
        self.formats = [".wav", '.mp3', ".mp4"]

        self.transcriber = AudioTranscriber(
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        if path.suffix not in self.formats:
            logger.warning(
                f'This audio/video file ({path.suffix}) is not supported.'
                f'The format should be {self.formats}'
            )
            return None

        # load it in wave format
        if path.suffix == '.wav':
            audio_path_wav = path
        else:
            sound = AudioSegment.from_file(file=path, format=path.suffix[1:])
            audio_path_wav = path.with_suffix('.wav')
            # Export to wav format
            sound.export(
                audio_path_wav, format="wav", 
                parameters=["-ar", "16000", "-ac", "1", "-ab", "32k"]
            )
        
        audio = whisperx.load_audio(audio_path_wav)

        if path.suffix != '.wav':
            os.remove(audio_path_wav)

        transcription_l = self.transcriber.model.transcribe(audio, batch_size=self.batch_size)
        content = ' '.join([tr['text'] for tr in transcription_l['segments']])

        self.free_memory()

        return Document(
            page_content=f"{content}{self.page_sep}",
            metadata=metadata
        )