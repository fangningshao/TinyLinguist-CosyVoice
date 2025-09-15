import os
import datetime as dt
import logging

# Configure logging to WARNING level to reduce verbose output
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce verbosity of noisy libraries
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.WARNING)
logging.getLogger('gradio').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_webdemo_dir():
    """
    Get the webdemo directory with current date format: ./webdemo_files/{YYYY-MM-DD}
    """
    date_str = dt.datetime.now().strftime("%Y-%m-%d")
    webdemo_dir = f"./webdemo_files/{date_str}"
    os.makedirs(webdemo_dir, exist_ok=True)
    return webdemo_dir


# Asset paths and model paths
ASSET_PATH = "./asset"
MODEL_PATH = './models'

LLM_CHECKPOINTS = {
    "Base Model": None,
    "Custom Model": f"{MODEL_PATH}/my_llm_model/epoch_5_whole.pt",
}

FLOW_CHECKPOINTS = {
    "Base Model": None,
    "Custom Model": f"{MODEL_PATH}/my_flow_model/epoch_5_whole.pt",
}


VOICE_TEMPLATES = [
    "yuqian1",
    "yuqian2",
    "degang1",
]

DEFAULT_LLM = "Base Model"
DEFAULT_FLOW = "Base Model"

GLOBAL_BASE_MODEL_PATH = "./models/CosyVoice2-0.5B/"
