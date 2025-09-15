import os
import socket
from typing import Optional

from config import ASSET_PATH, get_webdemo_dir
from temp_dirs import setup_gradio_temp_dir
from ui import create_interface


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            return local_ip
    except Exception:
        return "127.0.0.1"


def validate_ip_address(ip: str) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            sock.bind((ip, 0))
            return True
    except OSError as e:
        print(f"Warning: Cannot bind to IP {ip}: {e}")
        return False


def find_available_port(start_port=8600, end_port=8999, ip="127.0.0.1") -> int:
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.bind((ip, port))
                return port
        except OSError:
            continue
    raise OSError(f"No available ports found in range {start_port}-{end_port} for IP {ip}")


def launch_with_port_retry(demo, ip, start_port, end_port, max_retries=10) -> int:
    for attempt in range(max_retries):
        try:
            selected_port = find_available_port(start_port, end_port, ip)
            print(f"Attempt {attempt + 1}: Trying {ip}:{selected_port}")
            demo.launch(
                server_name=ip,
                server_port=selected_port,
                share=False,
                debug=True,
                show_error=True,
                prevent_thread_lock=False,
                quiet=False,
                allowed_paths=[ASSET_PATH, "./webdemo_files", "/tmp", os.getcwd()]
            )
            print(f"Successfully launched on {ip}:{selected_port}")
            return selected_port
        except OSError as e:
            if any(k in str(e).lower() for k in ["port", "bind", "address"]):
                print(f"Warning: {ip}:{selected_port} failed: {e}")
                start_port = selected_port + 1
                if start_port > end_port:
                    print(f"Exhausted port range {start_port}-{end_port}")
                    break
                continue
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            start_port += 1
            if start_port > end_port:
                break
            continue
    raise OSError(f"Failed to launch after {max_retries} attempts in range")


def main(ip="127.0.0.1", port: Optional[int] = None, port_range_start=8600, port_range_end=8999, skip_load=False):
    print("Starting CosyVoice2 Demo Server...")
    print(f"Target IP: {ip}")
    print(f"Port range: {port_range_start}-{port_range_end}")

    try:
        setup_gradio_temp_dir()
        get_webdemo_dir()
        print("Temp directories initialized successfully")
    except Exception as e:
        print(f"Warning: Error setting up temp directories: {e}")

    original_ip = ip
    if not validate_ip_address(ip):
        print(f"Cannot bind to IP address: {ip}")
        local_ip = get_local_ip()
        print(f"Detected local IP: {local_ip}")
        for alt_ip in [local_ip, "127.0.0.1", "0.0.0.0"]:
            if alt_ip != original_ip and validate_ip_address(alt_ip):
                print(f"Using alternative IP: {alt_ip}")
                ip = alt_ip
                break
        else:
            print("No valid IP address found. Available options:")
            print("   - Check if the specified IP is configured on this machine")
            print("   - Use 127.0.0.1 for local access only")
            print("   - Use 0.0.0.0 to bind to all interfaces")
            return
    else:
        print(f"IP address {ip} is valid")

    if not skip_load:
        from model_cache import handle_load_model_button
        from config import DEFAULT_LLM, DEFAULT_FLOW
        try:
            handle_load_model_button(DEFAULT_LLM, DEFAULT_FLOW)
        except Exception as e:
            print(f"Error loading default model: {e}")
            print("Will load model when user selects one...")
    else:
        print("Skipping default model loading - model will be loaded when user clicks 'Load Model'")

    demo = create_interface()
    extra_allowed = list({ASSET_PATH, "./webdemo_files", "/tmp", os.getcwd(), os.path.dirname(os.path.abspath(__file__))})

    if port is not None:
        try:
            print(f"Trying user-specified port: {ip}:{port}")
            demo.launch(server_name=ip, server_port=int(port), share=False, debug=True, show_error=True, allowed_paths=extra_allowed)
            print(f"Successfully launched on specified port {ip}:{port}")
            return
        except OSError as e:
            if any(k in str(e).lower() for k in ["port", "bind", "address"]):
                print(f"Warning: Specified port {ip}:{port} unavailable: {e}")
                print(f"Falling back to port range {port_range_start}-{port_range_end}")
            else:
                raise e

    print(f"Launching web interface on {ip} with port range {port_range_start}-{port_range_end}...")
    try:
        launch_with_port_retry(demo, ip, port_range_start, port_range_end)
    except Exception as e:
        print(f"Failed to launch server: {e}")
        print("Troubleshooting suggestions:")
        print(f"   - Try using localhost: python webdemo_cv.py --ip=127.0.0.1")
        print(f"   - Try binding to all interfaces: python webdemo_cv.py --ip=0.0.0.0")
        print(f"   - Check if IP {original_ip} is properly configured on this machine")
        print(f"   - Verify firewall settings for ports {port_range_start}-{port_range_end}")
        return


