"""
Verify Wav2Vec2 Feature Extraction Dependencies
Checks if all required packages are installed for GPU-accelerated feature extraction
"""

import sys

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*70)
    print("🔍 Checking Wav2Vec2 Feature Extraction Dependencies")
    print("="*70 + "\n")
    
    missing_packages = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch: NOT INSTALLED")
        missing_packages.append("torch")
    
    # Check torchaudio
    try:
        import torchaudio
        print(f"✅ torchaudio: {torchaudio.__version__}")
    except ImportError:
        print("❌ torchaudio: NOT INSTALLED")
        missing_packages.append("torchaudio")
    
    # Check transformers
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers (Hugging Face): NOT INSTALLED")
        missing_packages.append("transformers")
    
    # Check numpy
    try:
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
    except ImportError:
        print("❌ numpy: NOT INSTALLED")
        missing_packages.append("numpy")
    
    # Check tqdm
    try:
        import tqdm
        print(f"✅ tqdm: {tqdm.__version__}")
    except ImportError:
        print("❌ tqdm: NOT INSTALLED")
        missing_packages.append("tqdm")
    
    print("\n" + "="*70)
    
    if missing_packages:
        print("\n❌ Missing packages detected!")
        print(f"   Install with: pip install {' '.join(missing_packages)}")
        print("\nFor GPU support:")
        print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   pip install transformers numpy tqdm")
        return False
    else:
        print("✅ All dependencies installed!")
        return True

if __name__ == "__main__":
    if check_dependencies():
        sys.exit(0)
    else:
        sys.exit(1)
