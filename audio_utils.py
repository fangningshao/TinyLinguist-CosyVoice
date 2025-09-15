import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple


def clean_line_for_tts(line: str) -> str:
    return line.replace("there's", "theirs").replace("There's", "Theirs")


def post_process_audio(
    audio_tensor: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: Optional[int] = None,
    target_bit_depth: Optional[int] = None,
) -> Tuple[torch.Tensor, int, Optional[str]]:
    """Post-process audio with sample rate conversion and bit depth adjustment.

    Returns: (processed_audio, final_sample_rate, encoding)
    """
    processed_audio = audio_tensor.clone()
    final_sample_rate = original_sample_rate
    encoding: Optional[str] = None

    if target_sample_rate is not None and target_sample_rate != original_sample_rate:
        print(
            f"Converting sample rate from {original_sample_rate}Hz to {target_sample_rate}Hz")
        resampler = T.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate)
        processed_audio = resampler(processed_audio)
        final_sample_rate = target_sample_rate

    if target_bit_depth is not None and target_bit_depth != 32:
        if target_bit_depth == 16:
            print("Converting to 16-bit depth")
            processed_audio = torch.clamp(processed_audio, -1.0, 1.0)
            encoding = "PCM_S"
        else:
            print(
                f"Warning: Unsupported bit depth {target_bit_depth}, keeping 32-bit")

    return processed_audio, final_sample_rate, encoding
