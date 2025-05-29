import modal

app = modal.App("lerobot-example")

# Create image with Python 3.10, ffmpeg, and LeRobot
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system dependencies including ffmpeg
    .apt_install("git", "ffmpeg")
    # Clone and install LeRobot
    .run_commands(
        "git clone https://github.com/huggingface/lerobot.git /lerobot",
        "cd /lerobot && pip install -e ."
    )
)

@app.function(gpu="H100", image=image)
def test_lerobot_setup():
    """Test that LeRobot is properly installed and can import"""
    import sys
    print(f"Python version: {sys.version}")
    
    # Test LeRobot imports
    try:
        import lerobot
        print(f"✅ LeRobot installed successfully, version: {lerobot.__version__}")
    except ImportError as e:
        print(f"❌ LeRobot import failed: {e}")
        return
    
    # Test PyTorch availability
    try:
        import torch
        print(f"✅ PyTorch available, version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not available")
    
    # Test ffmpeg
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg available")
        else:
            print("❌ FFmpeg not working")
    except FileNotFoundError:
        print("❌ FFmpeg not found")

@app.function(gpu="H100", image=image)
def run_lerobot_example():
    """Example function using LeRobot"""
    import torch
    import lerobot
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    print("Running LeRobot example...")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # You can add your LeRobot-specific code here
    # For example, loading a dataset or running a model
    
    print("LeRobot setup complete!")

@app.local_entrypoint()
def main():
    # Test the setup first
    test_lerobot_setup.remote()
    
    # Run your LeRobot code
    run_lerobot_example.remote()