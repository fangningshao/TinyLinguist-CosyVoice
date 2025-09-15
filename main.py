from server import main

if __name__ == "__main__":
    import argh
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    argh.dispatch_command(main)
