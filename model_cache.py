import datetime as dt
from typing import Optional, Tuple, List

import sys

from config import LLM_CHECKPOINTS, FLOW_CHECKPOINTS, GLOBAL_BASE_MODEL_PATH

# Add CosyVoice to path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2


global_cosyvoice_lrucache: List[Tuple[str, CosyVoice2]] = []
global_cache_size: int = 5
global_current_llm_ckpt: Optional[str] = None
global_current_flow_ckpt: Optional[str] = None


def _load_cosyvoice_model(llm_ckpt_name: str, flow_ckpt_name: str) -> CosyVoice2:
    llm_ckpt = LLM_CHECKPOINTS.get(llm_ckpt_name)
    flow_ckpt = FLOW_CHECKPOINTS.get(flow_ckpt_name)

    print(f"Loading CosyVoice2 with LLM: {llm_ckpt_name}, Flow: {flow_ckpt_name}")
    start_time = dt.datetime.now()

    this_cosyvoice = CosyVoice2(
        GLOBAL_BASE_MODEL_PATH,
        load_jit=False,
        load_trt=False,
        fp16=False,
        use_flow_cache=False,
        strict_load=False,
        flow_ckpt=flow_ckpt,
        llm_ckpt=llm_ckpt
    )

    load_time = (dt.datetime.now() - start_time).total_seconds()
    print(f"CosyVoice2 loaded successfully in {load_time:.2f}s\nLLM: {llm_ckpt_name}\nFlow: {flow_ckpt_name}")
    return this_cosyvoice


def get_model_from_cache(llm_ckpt_name: str, flow_ckpt_name: str) -> CosyVoice2:
    global global_cosyvoice_lrucache, global_cache_size, global_current_llm_ckpt, global_current_flow_ckpt
    print("Current loaded models:", [l[0] for l in global_cosyvoice_lrucache])
    global_current_llm_ckpt = llm_ckpt_name
    global_current_flow_ckpt = flow_ckpt_name

    model_name = llm_ckpt_name + ' + ' + flow_ckpt_name
    for idx, (name, model) in enumerate(global_cosyvoice_lrucache):
        if name == model_name:
            if idx != 0:
                global_cosyvoice_lrucache = [(model_name, model)] + [p for p in global_cosyvoice_lrucache if p[0] != model_name]
            return model

    loaded_model = _load_cosyvoice_model(llm_ckpt_name, flow_ckpt_name)
    if len(global_cosyvoice_lrucache) < global_cache_size:
        global_cosyvoice_lrucache = [(model_name, loaded_model)] + global_cosyvoice_lrucache
    else:
        global_cosyvoice_lrucache = [(model_name, loaded_model)] + global_cosyvoice_lrucache[:-1]
    return loaded_model


def handle_load_model_button(llm_ckpt_name: str, flow_ckpt_name: str) -> str:
    start_time = dt.datetime.now()
    try:
        _ = get_model_from_cache(llm_ckpt_name, flow_ckpt_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error loading CosyVoice2 model: {str(e)}"

    load_time = (dt.datetime.now() - start_time).total_seconds()
    success_msg = (
        f"CosyVoice2 loaded successfully in {load_time:.2f}s\nLLM: {llm_ckpt_name}\nFlow: {flow_ckpt_name}\n"
        f"Existing loaded models in cache: {[l[0] for l in global_cosyvoice_lrucache]}"
    )
    return success_msg


