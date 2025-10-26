"""
GPU Setup Verification Script
Tests all GPU components and provides diagnostic information
"""

import sys

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("\n" + "="*70)
    print("🔍 Checking PyTorch Installation")
    print("="*70)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test GPU tensor operations
            x = torch.randn(1000, 1000).cuda()
            y = x @ x.T
            print(f"✅ GPU tensor operations: Working (test passed)")
            del x, y
            torch.cuda.empty_cache()
            
            return True
        else:
            print("\n⚠️  CUDA not available!")
            print("   Reasons:")
            print("   1. No NVIDIA GPU detected")
            print("   2. NVIDIA drivers not installed")
            print("   3. PyTorch installed without CUDA support")
            print("\n   Install CUDA-enabled PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def check_torchaudio():
    """Check torchaudio installation."""
    print("\n" + "="*70)
    print("🎵 Checking torchaudio")
    print("="*70)
    
    try:
        import torchaudio
        print(f"✅ torchaudio version: {torchaudio.__version__}")
        
        # Test resampler on GPU
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            resampler = torchaudio.transforms.Resample(44100, 16000).to(device)
            test_audio = torch.randn(1, 44100).to(device)
            result = resampler(test_audio)
            print(f"✅ GPU resampling: Working (test passed)")
            del resampler, test_audio, result
            torch.cuda.empty_cache()
        
        return True
    except ImportError:
        print("❌ torchaudio not installed!")
        print("   Install with: pip install torchaudio")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def check_deepfilternet():
    """Check DeepFilterNet 2 installation."""
    print("\n" + "="*70)
    print("🔊 Checking DeepFilterNet 2")
    print("="*70)
    
    try:
        import torch
        from df.enhance import init_df
        print(f"✅ DeepFilterNet 2 (df package) installed")
        
        # Try to initialize the model
        try:
            model, state, _ = init_df()
            print(f"✅ DeepFilterNet 2 model initialized successfully")
            print(f"   Model type: {type(model).__name__}")
            
            # Check if model can use GPU
            if torch.cuda.is_available():
                model = model.to('cuda')
                print(f"   ✅ Model moved to GPU successfully")
            else:
                print(f"   ⚠️  GPU not available, using CPU")
            
            return True
                
        except Exception as e:
            print(f"⚠️  Could not initialize DeepFilterNet 2 model: {str(e)}")
            return False
            
    except ImportError:
        print("❌ DeepFilterNet 2 not installed!")
        print("   Install with: pip install deepfilternet")
        print("   Or: pip install df")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def check_dependencies():
    """Check other dependencies."""
    print("\n" + "="*70)
    print("📦 Checking Other Dependencies")
    print("="*70)
    
    deps = {
        'numpy': 'NumPy',
        'tqdm': 'tqdm (progress bars)',
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_ok = False
    
    return all_ok


def check_audio_files():
    """Check if audio files exist."""
    print("\n" + "="*70)
    print("📁 Checking Audio Files")
    print("="*70)
    
    from pathlib import Path
    
    audio_dir = Path("audio_data/raw_data")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("**/*.mp3"))
        if audio_files:
            print(f"✅ Found {len(audio_files)} MP3 files in {audio_dir}")
            return True
        else:
            print(f"⚠️  No MP3 files found in {audio_dir}")
            print("   Add audio files to process")
            return False
    else:
        print(f"⚠️  Directory not found: {audio_dir}")
        print("   Create directory and add audio files")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("🔬 GPU Audio Processing - Setup Verification")
    print("="*70)
    
    results = {
        "PyTorch": check_pytorch(),
        "torchaudio": check_torchaudio(),
        "DeepFilterNet": check_deepfilternet(),
        "Dependencies": check_dependencies(),
        "Audio Files": check_audio_files()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📋 Verification Summary")
    print("="*70)
    
    for component, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {component}: {'PASS' if status else 'FAIL'}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All checks passed! Ready for GPU-accelerated processing.")
        print("\n🚀 Run: cd src && python audio_preprocessing.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n📚 See GPU_SETUP.md for detailed instructions")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
