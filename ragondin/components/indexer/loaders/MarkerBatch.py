import atexit
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import math
import traceback

import click
import torch.multiprocessing as mp
from tqdm import tqdm
import gc

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.converters.pdf import PdfConverter
from marker.logger import configure_logging
from marker.models import create_model_dict
from marker.output import output_exists, save_output
from marker.settings import settings

configure_logging()


def worker_init(model_dict):
    if model_dict is None:
        model_dict = create_model_dict()

    global model_refs
    model_refs = model_dict

    # Ensure we clean up the model references on exit
    atexit.register(worker_exit)


def worker_exit():
    global model_refs
    try:
        del model_refs
    except Exception:
        pass


def process_single_pdf(args):
    fpath, cli_options = args
    config_parser = ConfigParser(cli_options)

    out_folder = config_parser.get_output_folder(fpath)
    base_name = config_parser.get_base_filename(fpath)
    if cli_options.get("skip_existing") and output_exists(out_folder, base_name):
        return

    try:
        if cli_options.get("debug_print"):
            print(f"Converting {fpath}")

        converter = PdfConverter(
            artifact_dict=model_refs,
            config={
                "output_format": "markdown",
                "paginate_output": True,
            },
        )
        rendered = converter(fpath)
        out_folder = config_parser.get_output_folder(fpath)
        save_output(rendered, out_folder, base_name)
        if cli_options.get("debug_print"):
            print(f"Converted {fpath}")
        del rendered
        del converter
    except Exception as e:
        print(f"Error converting {fpath}: {e}")
        print(traceback.format_exc())
    finally:
        gc.collect()


def MarkerBatchConvert(
    in_folder: str,
    chunk_idx: int = 0,
    num_chunks: int = 1,
    workers: int = 5,
    skip_existing: bool = False,
    debug_print: bool = False,
    max_tasks_per_worker: int = 10,
    **kwargs,
):
    in_folder = os.path.abspath(in_folder)
    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder) if f.endswith(".pdf")]
    files = [f for f in files if os.path.isfile(f)]
    kwargs["output_dir"] = in_folder

    if len(files) == 0:
        print(f"No pdf files found in {in_folder}")
        return

    # Disable nested multiprocessing
    kwargs["disable_multiprocessing"] = True

    total_processes = min(len(files), workers)

    if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
        model_dict = None
    else:
        try:
            mp.set_start_method("spawn")  # Required for CUDA, forkserver doesn't work
        except RuntimeError:
            raise RuntimeError(
                "Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again."
            )
        model_dict = create_model_dict()
        for k, v in model_dict.items():
            v.model.share_memory()

    print(
        f"Converting {len(files)} pdfs in chunk {chunk_idx + 1}/{num_chunks} with {total_processes} processes"
    )
    task_args = [(f, kwargs) for f in files]

    with mp.Pool(
        processes=total_processes,
        initializer=worker_init,
        initargs=(model_dict,),
        maxtasksperchild=max_tasks_per_worker,
    ) as pool:
        for _ in pool.imap_unordered(process_single_pdf, task_args):
            pass

    # Delete all CUDA tensors
    del model_dict
