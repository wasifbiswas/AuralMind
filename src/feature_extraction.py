"""
GPU-Accelerated Feature Extraction Pipeline
Extracts embeddings from audio files using Wav2Vec2-large-xlsr-53 on RTX 4060
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm
import time
import warnings
from transformers import Wav2Vec2FeatureExtractor as HFWav2Vec2FeatureExtractor, Wav2Vec2Model

warnings.filterwarnings('ignore')

# ============================================================================
# GPU Configuration
# ============================================================================

def check_gpu():
    """Verify GPU availability and display information."""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. GPU required for this pipeline.")
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("\n" + "="*70)
    print("🚀 GPU Acceleration Status")
    print("="*70)
    print(f"Device: cuda:0")
    print(f"GPU: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print("="*70 + "\n")
    
    return device

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e6  # MB
        reserved = torch.cuda.memory_reserved(0) / 1e6    # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1e6  # MB
        return allocated, reserved, total
    return 0, 0, 0

def print_gpu_stats(prefix=""):
    """Print current GPU memory statistics."""
    allocated, reserved, total = get_gpu_memory_usage()
    utilization = (allocated / total) * 100 if total > 0 else 0
    print(f"{prefix}[GPU: {utilization:.1f}%, Memory: {allocated:.0f}MB/{total:.0f}MB]")

# ============================================================================
# Audio Loading and Preprocessing
# ============================================================================

def load_audio_file(file_path: str, target_sr: int = 16000, device: str = 'cuda') -> Optional[torch.Tensor]:
    """
    Load audio file (MP3 or WAV) and convert to mono 16kHz waveform on GPU.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (16000 for Wav2Vec2)
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Audio tensor on GPU or None if loading fails
    """
    try:
        # Load audio using torchaudio (supports MP3 via ffmpeg backend)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
        
        # Move to GPU and squeeze to 1D
        waveform = waveform.squeeze(0).to(device)
        
        return waveform
        
    except Exception as e:
        print(f"  ❌ Error loading {os.path.basename(file_path)}: {str(e)}")
        return None

# ============================================================================
# Wav2Vec2 Model Setup
# ============================================================================

class Wav2Vec2FeatureExtractor:
    """GPU-accelerated feature extraction using Wav2Vec2-large-xlsr-53."""
    
    def __init__(self, device: torch.device):
        """
        Initialize Wav2Vec2 model and processor on GPU.
        
        Args:
            device: PyTorch device (cuda:0)
        """
        self.device = device
        self.model_name = "facebook/wav2vec2-large-xlsr-53"
        
        print("🔄 Loading Wav2Vec2-large-xlsr-53 model...")
        start_time = time.time()
        
        # Load feature extractor (for audio preprocessing)
        self.feature_extractor = HFWav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        
        # Load model and move to GPU (use safetensors to avoid PyTorch version requirement)
        self.model = Wav2Vec2Model.from_pretrained(
            self.model_name,
            use_safetensors=True  # Force use of safetensors format
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        elapsed = time.time() - start_time
        print(f"✅ Model loaded on GPU in {elapsed:.2f}s")
        print_gpu_stats("   ")
        
    def extract_features(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract feature embeddings from audio waveform.
        
        Args:
            waveform: Audio tensor on GPU (1D)
            
        Returns:
            Feature embeddings (last_hidden_state) or None if extraction fails
        """
        try:
            # Move waveform to CPU for feature extractor
            waveform_cpu = waveform.cpu().numpy()
            
            # Process audio through feature extractor
            inputs = self.feature_extractor(
                waveform_cpu,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move input tensors to GPU
            input_values = inputs.input_values.to(self.device)
            
            # Extract features (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(input_values)
                
            # Get last hidden state (main feature embeddings)
            # Shape: (batch_size, sequence_length, hidden_size=1024)
            features = outputs.last_hidden_state
            
            # Squeeze batch dimension and move to CPU for saving
            features = features.squeeze(0).cpu()
            
            return features
            
        except Exception as e:
            print(f"  ❌ Error extracting features: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up GPU memory."""
        del self.model
        del self.feature_extractor
        torch.cuda.empty_cache()

# ============================================================================
# Batch Processing Pipeline
# ============================================================================

def batch_extract_features(
    input_dir: str,
    output_dir: str,
    device: torch.device,
    file_extensions: List[str] = ['.wav', '.mp3']
) -> dict:
    """
    Batch process audio files and extract Wav2Vec2 features.
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save feature embeddings
        device: PyTorch device (cuda:0)
        file_extensions: List of supported audio file extensions
        
    Returns:
        Dictionary with processing statistics
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        return {"error": "Input directory not found"}
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(list(input_path.glob(f"**/*{ext}")))
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return {"error": "No audio files found"}
    
    print("\n" + "="*70)
    print("🎵 GPU-Accelerated Feature Extraction Pipeline")
    print("="*70)
    print(f"Model: facebook/wav2vec2-large-xlsr-53")
    print(f"Device: {device}")
    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Total files found: {len(audio_files)}")
    print("="*70 + "\n")
    
    # Initialize feature extractor
    extractor = Wav2Vec2FeatureExtractor(device)
    
    # Statistics tracking
    stats = {
        "total_files": len(audio_files),
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "total_duration": 0.0,
        "processing_times": [],
        "feature_shapes": []
    }
    
    # Start timing
    overall_start = time.time()
    
    # Process each file with progress bar
    with tqdm(total=len(audio_files), desc="🎵 Extracting features", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for audio_file in audio_files:
            file_start = time.time()
            
            # Get relative path to maintain folder structure
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.npy')
            
            # Skip if already processed
            if output_file.exists():
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "⏭️ exists", "file": relative_path.name[:30]})
                continue
            
            # Load audio file
            waveform = load_audio_file(str(audio_file), device=device)
            if waveform is None:
                stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "❌ load failed", "file": relative_path.name[:30]})
                continue
            
            # Check for silence
            if torch.max(torch.abs(waveform)).item() < 0.001:
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "🔇 silent", "file": relative_path.name[:30]})
                continue
            
            # Extract features
            features = extractor.extract_features(waveform)
            if features is None:
                stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "❌ extraction failed", "file": relative_path.name[:30]})
                continue
            
            # Save features
            output_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.save(str(output_file), features.numpy())
                
                # Calculate metrics
                file_time = time.time() - file_start
                stats["processed"] += 1
                stats["processing_times"].append(file_time)
                stats["feature_shapes"].append(features.shape)
                
                # Get GPU stats
                allocated, _, total = get_gpu_memory_usage()
                gpu_util = (allocated / total) * 100
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "status": "✅ done",
                    "shape": f"{features.shape}",
                    "GPU": f"{gpu_util:.0f}%",
                    "mem": f"{allocated:.0f}MB",
                    "file": relative_path.name[:25]
                })
                
            except Exception as e:
                print(f"\n  ❌ Error saving {relative_path.name}: {str(e)}")
                stats["failed"] += 1
                pbar.update(1)
    
    # Calculate final metrics
    total_time = time.time() - overall_start
    avg_time = np.mean(stats["processing_times"]) if stats["processing_times"] else 0
    
    # Clean up
    print("\n🧹 Cleaning up GPU resources...")
    extractor.cleanup()
    print("✅ GPU memory cleared")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Feature Extraction Summary")
    print("="*70)
    print(f"✅ Successfully processed: {stats['processed']}/{stats['total_files']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes")
    print(f"⚡ Average time per file: {avg_time:.3f}s")
    print(f"🚀 Processing speed: {stats['processed']/total_time:.2f} files/sec")
    
    if stats["feature_shapes"]:
        sample_shape = stats["feature_shapes"][0]
        print(f"📦 Feature shape example: {sample_shape}")
        print(f"   (sequence_length={sample_shape[0]}, hidden_size={sample_shape[1]})")
    
    print(f"💾 Output directory: {output_path}")
    print("="*70 + "\n")
    
    return stats

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Check GPU availability
        device = check_gpu()
        
        # Define paths - adjust these to your needs
        input_dir = "audio_data/clean_audio"  # Use cleaned audio from preprocessing
        output_dir = "audio_data/features"
        
        print("\n" + "="*70)
        print("🎙️  Wav2Vec2 Feature Extraction Pipeline")
        print("   GPU-Accelerated Embedding Generation")
        print("="*70 + "\n")
        
        # Run batch feature extraction
        stats = batch_extract_features(
            input_dir=input_dir,
            output_dir=output_dir,
            device=device
        )
        
        if "error" not in stats:
            print("✅ Feature extraction completed successfully!")
            print(f"   Processed {stats['processed']} files using RTX 4060 8GB GPU")
        else:
            print(f"❌ Error: {stats['error']}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
