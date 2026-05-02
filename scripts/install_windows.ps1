# One-time setup for the Windows lab machine with NVIDIA GPU.
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
#
# Adjust the CUDA tag (cu121 / cu124 / cu126) to match your driver:
#   nvidia-smi   -> shows your CUDA version

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment .venv ..."
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing torch with CUDA 12.1 wheels..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing remaining requirements..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Verifying installation:"
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
