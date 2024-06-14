"""
Common utilities.
"""
import logging
import logging.handlers
import os
import platform
import sys
import warnings
from pathlib import Path
from uuid import uuid4
from huggingface_hub import CommitScheduler
import requests
import json


# build a local storage that will schedule uploads to the hub
# this is the robust way of pushing it, see https://huggingface.co/spaces/Wauplin/space_to_dataset_saver
JSON_DATASET_DIR = Path("results_dataset_to_upload")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Each instance of this space will spawn a unique file for each type of result
# For the life of that space, it will append to that file pushed to a dataset every so often
# It also is append_only, so no previous data will be overwritten
JSON_DATASET_PATH = JSON_DATASET_DIR / f"NAME_TO_REPLACE-{uuid4()}.jsonl"

scheduler = CommitScheduler(
    repo_id="mteb/arena-results",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="data",
    every=5,
    token=os.environ["HF_TOKEN"]
)

#from .utils import save_log_str_on_log_server

handler = None
visited_loggers = set()

LOGDIR = os.getenv("LOGDIR", "./MTEB-Arena-logs/vote_log")

class APIHandler(logging.Handler):
    """Custom logging handler that sends logs to an API."""

    def __init__(self, apiUrl, log_path, *args, **kwargs):
        super(APIHandler, self).__init__(*args, **kwargs)
        self.apiUrl = apiUrl
        self.log_path = log_path

    def emit(self, record):
        log_entry = self.format(record)
        try:
            save_log_str_on_log_server(log_entry, self.log_path)
        except requests.RequestException as e:
            print(f"Error sending log to API: {e}", file=sys.stderr)

def build_logger(logger_name, logger_filename, add_remote_handler=False):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        if sys.version_info[1] >= 9:
            # This is for windows
            logging.basicConfig(level=logging.INFO, encoding="utf-8")
        else:
            if platform.system() == "Windows":
                warnings.warn(
                    "If you are running on Windows, "
                    "we recommend you use Python >= 3.9 for UTF-8 encoding."
                )
            logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if add_remote_handler:
        # Add APIHandler to send logs to your API
        api_url = f"{LOG_SERVER_ADDR}/{SAVE_LOG}"
        
        remote_logger_filename = str(Path(logger_filename).stem + "_remote.log")
        api_handler = APIHandler(apiUrl=api_url, log_path=f"{LOGDIR}/{remote_logger_filename}")
        api_handler.setFormatter(formatter)
        logger.addHandler(api_handler)
        
        stdout_logger.addHandler(api_handler)
        stderr_logger.addHandler(api_handler)

    # if LOGDIR is empty, then don't try output log to local file
    if LOGDIR != "":
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="utf-8"
        )
        handler.setFormatter(formatter)

        for l in [stdout_logger, stderr_logger, logger]:
            if l in visited_loggers:
                continue
            visited_loggers.add(l)
            l.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                encoded_message = line.encode("utf-8", "ignore").decode("utf-8")
                self.logger.log(self.log_level, encoded_message.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            encoded_message = self.linebuf.encode("utf-8", "ignore").decode("utf-8")
            self.logger.log(self.log_level, encoded_message.rstrip())
        self.linebuf = ""


def store_data_in_hub(message: str, message_type: str):
    with scheduler.lock:
        file_to_upload = Path(str(JSON_DATASET_PATH).replace("NAME_TO_REPLACE", message_type))
        with file_to_upload.open("a") as f:
            json.dump(message, f)
            f.write("\n")
