import os
import zipfile
import datetime as dt
from typing import Optional, Tuple, List

import torch
import torchaudio
import numpy as np
import gradio as gr

from cosyvoice.utils.file_utils import load_wav

from config import ASSET_PATH, DEFAULT_FLOW, DEFAULT_LLM, get_webdemo_dir
from audio_utils import post_process_audio, clean_line_for_tts
from model_cache import get_model_from_cache, global_current_llm_ckpt, global_current_flow_ckpt


def load_voice_template(voice_name: str):
    try:
        from pathlib import Path
        txt_path = Path(ASSET_PATH) / f"{voice_name}.txt"
        wav_path = Path(ASSET_PATH) / f"{voice_name}.wav"

        if not txt_path.exists():
            return None, f"Text file not found: {txt_path}", None
        if not wav_path.exists():
            return None, f"Audio file not found: {wav_path}", None

        with open(txt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()

        waveform, sample_rate = torchaudio.load(wav_path)
        duration = waveform.shape[1] / sample_rate
        print(f"Loaded template {voice_name}: {duration:.1f}s, {sample_rate}Hz")
        success_msg = f"Loaded {voice_name}\nDuration: {duration:.1f}s at {sample_rate}Hz"
        return str(wav_path), success_msg, prompt_text
    except Exception as e:
        error_msg = f"Error loading {voice_name}: {str(e)}"
        print(error_msg)
        return None, error_msg, None


def clone_voice_zero_shot_with_download(
    llm_ckpt_name: str,
    flow_ckpt_name: str,
    audio_input,
    prompt_text: str,
    target_text: str,
    speed: float = 1.0,
    trim_ending: int = 40,
    stream: bool = False,
    target_sample_rate: Optional[int] = None,
    target_bit_depth: Optional[int] = None,
) -> Tuple[Optional[str], str, Optional[str]]:
    output_path, status_msg = clone_voice_zero_shot(
        llm_ckpt_name, flow_ckpt_name, audio_input, prompt_text, target_text, speed, trim_ending, stream, target_sample_rate, target_bit_depth
    )
    return output_path, status_msg, output_path


def clone_voice_zero_shot(
    llm_ckpt_name: str,
    flow_ckpt_name: str,
    audio_input,
    prompt_text: str,
    target_text: str,
    speed: float = 1.0,
    trim_ending: int = 40,
    stream: bool = False,
    target_sample_rate: Optional[int] = None,
    target_bit_depth: Optional[int] = None,
) -> Tuple[Optional[str], str]:
    this_cosyvoice = get_model_from_cache(llm_ckpt_name, flow_ckpt_name)

    if not target_text.strip():
        return None, "Please provide target text to synthesize"
    if audio_input is None:
        return None, "Please record or upload an audio file"
    if not prompt_text.strip():
        return None, "Please provide the prompt text (what you said in the recording)"

    try:
        prompt_wav_path = None
        cleanup_temp = False

        if isinstance(audio_input, str):
            prompt_wav_path = audio_input
            print(f"Using prompt wav file: {prompt_wav_path}")
        else:
            try:
                webdemo_dir = get_webdemo_dir()
                temp_filename = f"recorded_audio_{dt.datetime.now().strftime('%H%M%S_%f')}.wav"
                temp_path = os.path.join(webdemo_dir, temp_filename)

                if isinstance(audio_input, tuple):
                    sample_rate, audio_array = audio_input
                    if isinstance(audio_array, np.ndarray):
                        waveform = torch.from_numpy(audio_array).float()
                    else:
                        waveform = torch.tensor(audio_array, dtype=torch.float32)
                    if len(waveform.shape) == 1:
                        waveform = waveform.unsqueeze(0)
                    elif len(waveform.shape) == 2 and waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if waveform.abs().max() > 1.0:
                        waveform = waveform / waveform.abs().max()
                    torchaudio.save(temp_path, waveform, sample_rate)
                    prompt_wav_path = temp_path
                    cleanup_temp = True
                    print(f"Saved recorded audio to: {prompt_wav_path}")
                else:
                    return None, f"Unsupported audio format: {type(audio_input)}"
            except Exception as audio_error:
                return None, f"Error processing recorded audio: {str(audio_error)}"

        if not os.path.exists(prompt_wav_path):
            return None, f"Audio file not found: {prompt_wav_path}"

        try:
            test_waveform, test_sr = torchaudio.load(prompt_wav_path)
            duration = test_waveform.shape[1] / test_sr
            print(f"Audio validation - Duration: {duration:.2f}s, Sample rate: {test_sr}Hz")
            if duration < 1.0:
                if cleanup_temp: os.unlink(prompt_wav_path)
                return None, f"Audio too short ({duration:.1f}s). Please record at least 1 second."
            elif duration > 60.0:
                if cleanup_temp: os.unlink(prompt_wav_path)
                return None, f"Audio too long ({duration:.1f}s). Please keep it under 60 seconds."
        except Exception as e:
            if cleanup_temp: os.unlink(prompt_wav_path)
            return None, f"Error validating audio file: {str(e)}"

        try:
            prompt_speech_16k = load_wav(prompt_wav_path, 16000)
            print(f"Loaded audio at 16kHz: {prompt_speech_16k.shape}")
        except Exception as e:
            if cleanup_temp: os.unlink(prompt_wav_path)
            return None, f"Error loading audio at 16kHz: {str(e)}"

        webdemo_dir = get_webdemo_dir()
        output_filename = f"output_zs_{dt.datetime.now().strftime('%H%M%S_%f')}.wav"
        output_path = os.path.join(webdemo_dir, output_filename)

        start_time = dt.datetime.now()
        print(f"Generating speech for: {target_text[:50]}...")
        print(f"Using prompt: {prompt_text[:50]}...")

        audio_segments: List[torch.Tensor] = []
        try:
            for i, j in enumerate(this_cosyvoice.inference_zero_shot(
                clean_line_for_tts(target_text),
                prompt_text,
                prompt_speech_16k,
                stream=stream,
                speed=speed
            )):
                audio_segments.append(j['tts_speech'])
                if i == 0:
                    trimmed_audio = j['tts_speech']
                    if trim_ending > 0:
                        trim_samples = int(trim_ending * this_cosyvoice.sample_rate / 1000)
                        if trimmed_audio.shape[1] > trim_samples * 2:
                            trimmed_audio = trimmed_audio[:, :-trim_samples]
                    processed_audio, final_sr, encoding = post_process_audio(
                        trimmed_audio,
                        this_cosyvoice.sample_rate,
                        target_sample_rate,
                        target_bit_depth
                    )
                    if encoding:
                        torchaudio.save(output_path, processed_audio, final_sr, encoding=encoding, bits_per_sample=16)
                    else:
                        torchaudio.save(output_path, processed_audio, final_sr)
                    print(f"Saved first segment: {j['tts_speech'].shape} -> {processed_audio.shape} at {final_sr}Hz")
        except Exception as e:
            if cleanup_temp: os.unlink(prompt_wav_path)
            return None, f"Error during TTS inference: {str(e)}"

        if len(audio_segments) > 1:
            try:
                final_audio = torch.cat(audio_segments, dim=1)
                if trim_ending > 0:
                    trim_samples = int(trim_ending * this_cosyvoice.sample_rate / 1000)
                    if final_audio.shape[1] > trim_samples * 2:
                        final_audio = final_audio[:, :-trim_samples]
                processed_audio, final_sr, encoding = post_process_audio(
                    final_audio,
                    this_cosyvoice.sample_rate,
                    target_sample_rate,
                    target_bit_depth
                )
                if encoding:
                    torchaudio.save(output_path, processed_audio, final_sr, encoding=encoding, bits_per_sample=16)
                else:
                    torchaudio.save(output_path, processed_audio, final_sr)
                print(f"Concatenated {len(audio_segments)} segments: {final_audio.shape} -> {processed_audio.shape} at {final_sr}Hz")
            except Exception as e:
                print(f"Warning: Error concatenating segments: {str(e)}")

        generation_time = (dt.datetime.now() - start_time).total_seconds()

        if audio_segments:
            total_samples = sum(seg.shape[1] for seg in audio_segments)
            audio_duration = total_samples / this_cosyvoice.sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else 0
        else:
            audio_duration = 0
            rtf = 0

        post_process_info = ""
        if target_sample_rate is not None or target_bit_depth is not None:
            post_process_info = "\nPost-processing applied:"
            if target_sample_rate is not None:
                post_process_info += f" Sample rate: {target_sample_rate}Hz"
            if target_bit_depth is not None:
                post_process_info += f" Bit depth: {target_bit_depth}-bit"

        trim_info = f"\nTrim ending: {trim_ending}ms" if trim_ending > 0 else ""

        success_msg = (
            f"Voice cloned successfully!\n"
            f"Generated {audio_duration:.1f}s audio in {generation_time:.1f}s\n"
            f"RTF: {rtf:.3f} (lower is faster)\n"
            f"Target text: {target_text[:50]}{'...' if len(target_text) > 50 else ''}\n"
            f"Files saved in: {get_webdemo_dir()}/\n"
            f"Using: Zero-shot inference{post_process_info}{trim_info}"
        )

        return output_path, success_msg
    except Exception as e:
        import traceback
        error_msg = f"Error during voice cloning: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def batch_inference(
    llm_ckpt_name: str,
    flow_ckpt_name: str,
    audio_input,
    prompt_text: str,
    text_file,
    speed: float = 1.0,
    trim_ending: int = 40,
    use_instruct2: bool = False,
    instruct_text: str = "用中文说这句话",
    target_sample_rate: Optional[int] = None,
    target_bit_depth: Optional[int] = None,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    this_cosyvoice = get_model_from_cache(llm_ckpt_name, flow_ckpt_name)
    if this_cosyvoice is None:
        return None, "Model not loaded. Please select and load a model first."
    if audio_input is None:
        return None, "Please record or upload an audio file"
    if text_file is None:
        return None, "Please upload a text file with sentences to synthesize"
    if not use_instruct2 and not prompt_text.strip():
        return None, "Please provide the prompt text for zero-shot mode"

    try:
        if isinstance(audio_input, str):
            prompt_wav_path = audio_input
            print(f"Using prompt wav file: {prompt_wav_path}")
            cleanup_temp = False
        else:
            webdemo_dir = get_webdemo_dir()
            temp_filename = f"recorded_audio_batch_{dt.datetime.now().strftime('%H%M%S_%f')}.wav"
            prompt_wav_path = os.path.join(webdemo_dir, temp_filename)
            cleanup_temp = True
            if isinstance(audio_input, tuple):
                sample_rate, audio_array = audio_input
                waveform = torch.from_numpy(audio_array).float()
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                elif len(waveform.shape) == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                torchaudio.save(prompt_wav_path, waveform, sample_rate)
            else:
                return None, "Invalid audio format"

        prompt_speech_16k = load_wav(prompt_wav_path, 16000)

        with open(text_file.name, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        if not raw_lines:
            return None, "No valid text lines found in the uploaded file"

        lines: List[str] = []
        filenames: List[Optional[str]] = []
        for raw_line in raw_lines:
            if '\t' in raw_line:
                parts = raw_line.split('\t', 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    text = parts[1].strip()
                    if filename and text:
                        filenames.append(filename)
                        lines.append(text)
            else:
                text = raw_line.strip()
                if text:
                    filenames.append(None)
                    lines.append(text)

        if not lines:
            return None, "No valid text lines found after parsing"

        print(f"Processing {len(lines)} lines of text...")
        if any(f is not None for f in filenames):
            print("Using custom filenames from first column")

        webdemo_dir = get_webdemo_dir()
        batch_subdir = f"batch_{dt.datetime.now().strftime('%H%M%S')}"
        temp_dir = os.path.join(webdemo_dir, batch_subdir)
        os.makedirs(temp_dir, exist_ok=True)
        audio_files: List[str] = []
        start_time = dt.datetime.now()
        successful_files = 0
        failed_files = 0
        error_log: List[str] = []
        file_mapping = {}

        for idx, line in enumerate(lines):
            try:
                progress((idx + 1) / len(lines), f"Processing {idx + 1}/{len(lines)}: {line[:30]}...")
                if not line.strip() or len(line.strip()) < 3:
                    print(f"Skipping line {idx}: too short or empty")
                    failed_files += 1
                    error_log.append(f"Line {idx}: Skipped (too short or empty)")
                    continue

                if filenames[idx] is not None:
                    custom_name = filenames[idx]
                    if not custom_name.lower().endswith('.wav'):
                        custom_name += '.wav'
                    output_path = os.path.join(temp_dir, custom_name)
                    file_mapping[custom_name] = line
                else:
                    output_path = os.path.join(temp_dir, f"{str(idx).zfill(5)}.wav")
                    file_mapping[f"{str(idx).zfill(5)}.wav"] = line

                def inference_with_timeout():
                    try:
                        if use_instruct2:
                            inference_generator = this_cosyvoice.inference_instruct2(
                                clean_line_for_tts(line),
                                instruct_text,
                                prompt_speech_16k,
                                stream=False,
                                speed=speed
                            )
                        else:
                            inference_generator = this_cosyvoice.inference_zero_shot(
                                clean_line_for_tts(line),
                                prompt_text,
                                prompt_speech_16k,
                                stream=False,
                                speed=speed
                            )
                        audio_segments_local = []
                        for i, j in enumerate(inference_generator):
                            if 'tts_speech' in j:
                                audio_segments_local.append(j['tts_speech'])
                            else:
                                print(f"Warning: No tts_speech in result {i} for line {idx}")
                        return audio_segments_local
                    except Exception as e:
                        raise e

                audio_segments = inference_with_timeout()
                if audio_segments:
                    if len(audio_segments) > 1:
                        final_audio = torch.cat(audio_segments, dim=1)
                    else:
                        final_audio = audio_segments[0]
                    if final_audio.shape[1] < 1000:
                        print(f"Warning: Very short audio for line {idx}: {final_audio.shape}")
                        failed_files += 1
                        error_log.append(f"Line {idx}: Audio too short ({final_audio.shape[1]} samples)")
                        continue
                    if trim_ending > 0:
                        trim_samples = int(trim_ending * this_cosyvoice.sample_rate / 1000)
                        if final_audio.shape[1] > trim_samples * 2:
                            final_audio = final_audio[:, :-trim_samples]
                    processed_audio, final_sr, encoding = post_process_audio(
                        final_audio,
                        this_cosyvoice.sample_rate,
                        target_sample_rate,
                        target_bit_depth
                    )
                    try:
                        if encoding:
                            torchaudio.save(output_path, processed_audio, final_sr, encoding=encoding, bits_per_sample=16)
                        else:
                            torchaudio.save(output_path, processed_audio, final_sr)
                        audio_files.append(output_path)
                        successful_files += 1
                        print(f"Generated: {os.path.basename(output_path)} - {line[:30]}...")
                    except Exception as save_error:
                        print(f"Error saving audio for line {idx}: {save_error}")
                        failed_files += 1
                        error_log.append(f"Line {idx}: Save error - {str(save_error)}")
                        filename_key = os.path.basename(output_path)
                        if filename_key in file_mapping:
                            del file_mapping[filename_key]
                else:
                    print(f"No audio generated for line {idx}")
                    failed_files += 1
                    error_log.append(f"Line {idx}: No audio generated")

                if idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as line_error:
                print(f"Unexpected error processing line {idx}: {line_error}")
                failed_files += 1
                error_log.append(f"Line {idx}: Unexpected error - {str(line_error)[:100]}")
                continue

        generation_time = (dt.datetime.now() - start_time).total_seconds()
        if audio_files:
            progress(1.0, "Creating ZIP file...")
            zip_filename = f"batch_output_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(get_webdemo_dir(), zip_filename)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for audio_file in audio_files:
                    try:
                        zipf.write(audio_file, os.path.basename(audio_file))
                    except Exception as zip_error:
                        print(f"Warning: Could not add {audio_file} to zip: {zip_error}")

                post_process_summary = ""
                if target_sample_rate is not None:
                    post_process_summary += f"Sample Rate: {target_sample_rate}Hz\n"
                else:
                    post_process_summary += "Sample Rate: Original (24000Hz)\n"
                if target_bit_depth is not None:
                    post_process_summary += f"Bit Depth: {target_bit_depth}-bit\n"
                else:
                    post_process_summary += "Bit Depth: Original (32-bit)\n"

                summary_content = f"""CosyVoice2 Batch Inference Summary
Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Files stored in: {temp_dir}
Total input lines: {len(lines)}
Successful files: {successful_files}
Failed files: {failed_files}
Success rate: {(successful_files/len(lines)*100):.1f}%
Model: LLM={global_current_llm_ckpt}, Flow={global_current_flow_ckpt}
Mode: {'Instruct2' if use_instruct2 else 'Zero-shot'}
Speed: {speed}
Ending Trim: {trim_ending}ms
{post_process_summary}Total processing time: {generation_time:.1f}s
Average time per successful file: {generation_time/successful_files:.2f}s
{'Instruction: ' + instruct_text if use_instruct2 else 'Prompt: ' + prompt_text[:100]}

Successfully generated files:
"""
                for audio_file in audio_files:
                    filename = os.path.basename(audio_file)
                    if filename in file_mapping:
                        line_text = file_mapping[filename]
                        summary_content += f"{filename}: {line_text[:100]}{'...' if len(line_text) > 100 else ''}\n"
                    else:
                        summary_content += f"{filename}: (mapping not found)\n"
                if error_log:
                    summary_content += f"\nErrors encountered:\n"
                    for error in error_log[:20]:
                        summary_content += f"{error}\n"
                    if len(error_log) > 20:
                        summary_content += f"... and {len(error_log) - 20} more errors\n"
                zipf.writestr("summary.txt", summary_content)

        if cleanup_temp and os.path.exists(prompt_wav_path):
            try:
                os.unlink(prompt_wav_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete prompt file: {cleanup_error}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if audio_files:
            post_process_info = ""
            if target_sample_rate is not None or target_bit_depth is not None:
                post_process_info = "\nPost-processing:"
                if target_sample_rate is not None:
                    post_process_info += f" {target_sample_rate}Hz"
                if target_bit_depth is not None:
                    post_process_info += f" {target_bit_depth}-bit"
            success_msg = (
                f"Batch inference completed!\n"
                f"Generated {successful_files}/{len(lines)} files ({(successful_files/len(lines)*100):.1f}% success)\n"
                f"Total time: {generation_time:.1f}s\n"
                f"Average per file: {generation_time/successful_files:.2f}s\n"
                f"Mode: {'Instruct2' if use_instruct2 else 'Zero-shot'}{post_process_info}\n"
                f"Files saved in: {get_webdemo_dir()}/\n"
                f"ZIP file ready for download"
                + (f"\nWarning: {failed_files} files failed - see summary.txt for details" if failed_files > 0 else "")
            )
            return zip_path, success_msg
        else:
            return None, f"All files failed to generate. Check logs for details.\nErrors: {'; '.join(error_log[:3])}"
    except Exception as e:
        import traceback
        error_msg = f"Critical error during batch inference: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except:
            pass
        return None, error_msg


