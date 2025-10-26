"""
Environment Checker - Detects current setup and provides installation guidance
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return None

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print("="*70)
    print("🎮 Checking NVIDIA GPU")
    print("="*70)
    
    output = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    
    if output:
        print(f"✅ NVIDIA GPU detected: {output}")
        
        # Try to detect CUDA version
        cuda_output = run_command("nvidia-smi")
        if cuda_output and "CUDA Version:" in cuda_output:
            import re
            match = re.search(r"CUDA Version: ([\d.]+)", cuda_output)
            if match:
                cuda_version = match.group(1)
                print(f"✅ CUDA Version: {cuda_version}")
                
                # Determine recommended PyTorch version
                major_version = int(cuda_version.split('.')[0])
                if major_version >= 12:
                    return "cu121"
                elif major_version == 11:
                    return "cu118"
        return "cu118"  # default
    else:
        print("❌ No NVIDIA GPU detected or drivers not installed")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return None

def check_python_version():
    """Check Python version."""
    print("\n" + "="*70)
    print("🐍 Checking Python Version")
    print("="*70)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Python version compatible with PyTorch")
        return True
    else:
        print("⚠️  Python version may not be compatible")
        print("   Recommended: Python 3.8 - 3.11")
        return False

def check_current_packages():
    """Check currently installed packages."""
    print("\n" + "="*70)
    print("📦 Checking Installed Packages")
    print("="*70)
    
    packages = {
        'torch': 'PyTorch',
        'torchaudio': 'torchaudio',
        'pvkoala': 'Picovoice Koala',
        'tqdm': 'tqdm',
        'numpy': 'NumPy'
    }
    
    installed = {}
    
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            installed[pkg] = version
            
            # Check if PyTorch has CUDA support
            if pkg == 'torch':
                cuda_available = mod.cuda.is_available()
                cuda_version = mod.version.cuda if hasattr(mod.version, 'cuda') else 'N/A'
                print(f"   CUDA available: {cuda_available}")
                if cuda_available:
                    print(f"   CUDA version in PyTorch: {cuda_version}")
                else:
                    print(f"   ⚠️  PyTorch installed WITHOUT CUDA support!")
                
        except ImportError:
            print(f"❌ {name}: Not installed")
    
    return installed

def generate_installation_commands(cuda_version):
    """Generate installation commands based on detected setup."""
    print("\n" + "="*70)
    print("🔧 Recommended Installation Commands")
    print("="*70)
    
    if cuda_version:
        print(f"\n💻 For your system (CUDA {cuda_version.replace('cu', '')}), run:\n")
        
        print("# Step 1: Uninstall existing PyTorch (if any)")
        print("pip uninstall -y torch torchvision torchaudio\n")
        
        print("# Step 2: Install PyTorch with CUDA support")
        if cuda_version == "cu121":
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n# Step 3: Install other dependencies")
        print("pip install pvkoala tqdm numpy")
        
        print("\n# Step 4: Verify installation")
        print("python verify_gpu.py")
        
    else:
        print("\n⚠️  No NVIDIA GPU detected. You can still run on CPU:\n")
        print("pip install torch torchvision torchaudio")
        print("pip install pvkoala tqdm numpy")
        print("\nNote: Processing will be slower without GPU acceleration.")

def main():
    print("\n" + "="*70)
    print("🔍 GPU Audio Processing - Environment Checker")
    print("="*70)
    
    # Check components
    python_ok = check_python_version()
    cuda_version = check_nvidia_gpu()
    installed = check_current_packages()
    
    # Generate recommendations
    generate_installation_commands(cuda_version)
    
    # Final advice
    print("\n" + "="*70)
    print("📋 Next Steps")
    print("="*70)
    
    if cuda_version and 'torch' not in installed:
        print("\n1. Run the installation commands above")
        print("2. Verify with: python verify_gpu.py")
        print("3. Process audio: cd src && python audio_preprocessing.py")
    elif cuda_version and 'torch' in installed:
        try:
            import torch
            if torch.cuda.is_available():
                print("\n✅ Your environment is ready!")
                print("   Run: cd src && python audio_preprocessing.py")
            else:
                print("\n⚠️  PyTorch installed without CUDA support")
                print("   Reinstall using commands above for GPU acceleration")
        except:
            pass
    else:
        print("\n⚠️  No GPU detected - will run on CPU (slower)")
        print("   Run: cd src && python audio_preprocessing.py")
    
    print("\n📚 Documentation:")
    print("   - Quick Start: QUICK_START.md")
    print("   - Setup Guide: GPU_SETUP.md")
    print("   - What Changed: TRANSFORMATION_SUMMARY.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
