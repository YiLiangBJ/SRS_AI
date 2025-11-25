#!/usr/bin/env python3
"""
Installation and Environment Setup Script for SRS Channel Estimation

This script helps set up the Python environment on different platforms:
- Windows PC
- Linux servers  
- GPU clusters

Usage:
    python setup_environment.py [--gpu] [--intel-proxy]

Options:
    --gpu: Install GPU-specific packages
    --intel-proxy: Use Intel proxy for package installation
    --dev: Install development dependencies
"""

import os
import sys
import subprocess
import platform
import argparse
from typing import List, Optional


def run_command(cmd: str, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    print(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(
            cmd.split() if not shell else cmd,
            capture_output=True,
            text=True,
            check=check,
            shell=shell
        )
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"❌ stderr: {e.stderr}")
        if check:
            raise
        return e


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print("✅ Python version is compatible")


def detect_platform():
    """Detect the platform and hardware"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🖥️  Platform: {system} ({machine})")
    
    # Check for GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            print(f"🎮 GPU: {gpu_count} CUDA device(s) available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_properties(i).name
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("🔄 GPU: No CUDA devices available")
    except ImportError:
        print("🔄 GPU: PyTorch not installed yet")
    
    return system, gpu_available


def install_pytorch(use_gpu: bool = False, intel_proxy: bool = False):
    """Install PyTorch with appropriate configuration"""
    print(f"\n📦 Installing PyTorch (GPU: {use_gpu})...")
    
    # Determine PyTorch installation command
    if use_gpu:
        # Install GPU version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # Install CPU version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    # Add proxy if needed
    if intel_proxy:
        cmd += " --proxy http://child-prc.intel.com:913"
    
    run_command(cmd, shell=True)
    print("✅ PyTorch installation completed")


def install_requirements(intel_proxy: bool = False, dev: bool = False):
    """Install requirements from requirements.txt"""
    print(f"\n📦 Installing requirements...")
    
    # Base requirements
    cmd = "pip install -r requirements.txt"
    
    # Add proxy if needed
    if intel_proxy:
        cmd += " --proxy http://child-prc.intel.com:913"
    
    run_command(cmd, shell=True)
    
    # Install development dependencies
    if dev:
        print("📦 Installing development dependencies...")
        dev_packages = [
            "jupyterlab",
            "ipywidgets",
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy"
        ]
        
        for package in dev_packages:
            cmd = f"pip install {package}"
            if intel_proxy:
                cmd += " --proxy http://child-prc.intel.com:913"
            run_command(cmd, shell=True)
    
    print("✅ Requirements installation completed")


def install_sionna(intel_proxy: bool = False):
    """Install SIONNA for professional channel models"""
    print(f"\n📦 Installing SIONNA...")
    
    # Install TensorFlow first
    cmd = "pip install tensorflow>=2.13.0"
    if intel_proxy:
        cmd += " --proxy http://child-prc.intel.com:913"
    run_command(cmd, shell=True)
    
    # Install SIONNA
    cmd = "pip install sionna"
    if intel_proxy:
        cmd += " --proxy http://child-prc.intel.com:913"
    run_command(cmd, shell=True)
    
    print("✅ SIONNA installation completed")


def create_conda_env(env_name: str = "srs_ai", python_version: str = "3.9"):
    """Create a conda environment for the project"""
    print(f"\n🐍 Creating conda environment: {env_name}")
    
    # Check if conda is available
    try:
        run_command("conda --version")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Conda not available, skipping environment creation")
        return False
    
    # Create environment
    cmd = f"conda create -n {env_name} python={python_version} -y"
    run_command(cmd, shell=True)
    
    print(f"✅ Conda environment '{env_name}' created")
    print(f"💡 To activate: conda activate {env_name}")
    
    return True


def verify_installation():
    """Verify that all packages are installed correctly"""
    print(f"\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("tensorboard", "TensorBoard")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: Failed - {e}")
            failed_imports.append(name)
    
    # Test optional imports
    optional_imports = [
        ("sionna", "SIONNA"),
        ("tensorflow", "TensorFlow"),
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name}: OK (optional)")
        except ImportError:
            print(f"⚠️  {name}: Not installed (optional)")
    
    # Test CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: OK ({torch.cuda.device_count()} devices)")
        else:
            print(f"⚠️  CUDA: Not available")
    except ImportError:
        print(f"❌ CUDA: Cannot test (PyTorch not installed)")
    
    if failed_imports:
        print(f"\n❌ Installation verification failed!")
        print(f"Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print(f"\n✅ Installation verification successful!")
        return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup SRS Channel Estimation Environment")
    parser.add_argument('--gpu', action='store_true', help='Install GPU-specific packages')
    parser.add_argument('--intel-proxy', action='store_true', help='Use Intel proxy for installation')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    parser.add_argument('--create-env', type=str, help='Create conda environment with given name')
    parser.add_argument('--python-version', type=str, default='3.9', help='Python version for conda env')
    
    args = parser.parse_args()
    
    print("🚀 SRS Channel Estimation Environment Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Detect platform
    system, gpu_detected = detect_platform()
    
    # Create conda environment if requested
    if args.create_env:
        create_conda_env(args.create_env, args.python_version)
        print(f"\n💡 Please activate the environment and run this script again:")
        print(f"   conda activate {args.create_env}")
        print(f"   python setup_environment.py --gpu")
        return
    
    # Use GPU if detected and requested
    use_gpu = args.gpu or (gpu_detected and not args.gpu)
    
    try:
        # Install PyTorch
        install_pytorch(use_gpu=use_gpu, intel_proxy=args.intel_proxy)
        
        # Install requirements
        install_requirements(intel_proxy=args.intel_proxy, dev=args.dev)
        
        # Install SIONNA
        install_sionna(intel_proxy=args.intel_proxy)
        
        # Verify installation
        if verify_installation():
            print("\n🎉 Environment setup completed successfully!")
            print("\nNext steps:")
            print("1. Test the installation: python -c 'import torch; print(torch.__version__)'")
            print("2. Run system detection: python system_detection.py")
            print("3. Start training: python train_distributed.py")
            
            if use_gpu:
                print("4. For distributed training: python train_distributed.py --enable-ddp")
        else:
            print("\n❌ Environment setup failed!")
            print("Please check the error messages above and retry.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
