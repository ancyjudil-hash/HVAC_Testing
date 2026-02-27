# fix_pytorch.py
import subprocess
import sys

print("Fixing PyTorch installation...")

# Uninstall everything
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "transformers", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "langchain-text-splitters", "-y"])

# Install compatible versions
print("Installing compatible packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.30.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-text-splitters==0.0.1"])

print("Done! Please restart your application.")