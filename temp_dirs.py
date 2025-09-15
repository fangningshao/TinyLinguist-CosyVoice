import os
from typing import Optional


def setup_gradio_temp_dir() -> str:
    """
    Configure Gradio to use a writable temporary directory instead of /tmp/gradio/
    Returns the absolute path to the temp dir.
    """
    gradio_temp_dir = "./webdemo_files/gradio_temp"
    os.makedirs(gradio_temp_dir, exist_ok=True)

    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(gradio_temp_dir)

    if not os.access('/tmp', os.W_OK):
        absdir = os.path.abspath(gradio_temp_dir)
        os.environ["TMPDIR"] = absdir
        os.environ["TMP"] = absdir
        os.environ["TEMP"] = absdir

    try:
        vibe_edit_dir = os.path.join(gradio_temp_dir, "vibe_edit_history")
        os.makedirs(vibe_edit_dir, exist_ok=True)
        print(f"Created vibe_edit_history directory: {vibe_edit_dir}")
    except Exception as e:
        print(f"Warning: Could not create vibe_edit_history directory: {e}")

    print(f"Gradio temp directory set to: {gradio_temp_dir}")
    return gradio_temp_dir


