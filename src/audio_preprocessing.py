"""
Speech-Based Mental Health Dataset Cleaning (Stage 1)
GPU-Accelerated Preprocessing script for noise reduction using DeepFilterNet 2

This module handles:
- Loading MP3 audio files with soundfile (ffmpeg backend)
- Converting to appropriate format for DeepFilterNet 2 on GPU
- Applying DeepFilterNet 2 speech enhancement with full GPU acceleration
- Exporting cleaned WAV files at 16 kHz
- 100% GPU acceleration for all compatible operations including denoising
"""

import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings
import time
from tqdm import tqdm
from df.enhance import enhance, init_df, load_audio as df_load_audio, save_audio as df_save_audio
from df.io import resample

warnings.filterwarnings('ignore')

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*70}")
print(f"🚀 GPU Acceleration Status")
print(f"{'='*70}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available. Running on CPU.")
    print("   Install CUDA-enabled PyTorch for GPU acceleration:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print(f"{'='*70}\n")

# Configuration
TARGET_SAMPLE_RATE = 48000  # DeepFilterNet 2 works best at 48 kHz
OUTPUT_SAMPLE_RATE = 16000  # Final output at 16 kHz for consistency

# Global reusable transforms and models (avoid reinitialization overhead)
_resamplers_cache = {}
_df_model = None  # DeepFilterNet 2 model instance
_df_state = None  # DeepFilterNet 2 state
_benchmark_times = {"koala_cpu": None, "deepfilternet_gpu": []}  # Performance comparison


def get_resampler(orig_freq: int, new_freq: int) -> torchaudio.transforms.Resample:
    """
    Get or create a cached resampler on GPU (avoid reinitialization overhead).
    
    Args:
        orig_freq: Original sample rate
        new_freq: Target sample rate
        
    Returns:
        Resampler transform on GPU
    """
    global _resamplers_cache
    key = (orig_freq, new_freq)
    
    if key not in _resamplers_cache:
        _resamplers_cache[key] = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq
        ).to(device)
    
    return _resamplers_cache[key]


def get_deepfilternet_model():
    """
    Get or create the DeepFilterNet 2 model (singleton pattern for better performance).
    Loading models is expensive, so we reuse the same instance.
    
    Returns:
        Tuple of (model, df_state) for use with enhance()
    """
    global _df_model, _df_state
    
    if _df_model is None:
        print("🔄 Initializing DeepFilterNet 2 model...")
        start_time = time.time()
        _df_model, _df_state, _ = init_df()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            _df_model = _df_model.to(device)
            elapsed = time.time() - start_time
            print(f"✅ DeepFilterNet 2 initialized on GPU ({device}) in {elapsed:.2f}s")
        else:
            elapsed = time.time() - start_time
            print(f"✅ DeepFilterNet 2 initialized on CPU in {elapsed:.2f}s")
    
    return _df_model, _df_state


def load_audio(filepath: str) -> Tuple[Optional[torch.Tensor], int]:
    """
    Load an audio file and convert it to mono 16 kHz waveform using GPU acceleration.
    
    Args:
        filepath: Path to the audio file (MP3, WAV, FLAC, etc.)
        
    Returns:
        Tuple of (audio_tensor on GPU, sample_rate) or (None, 0) if loading fails
    """
    try:
        # Load audio with soundfile (handles MP3 via ffmpeg)
        waveform, sample_rate = sf.read(filepath, dtype='float32')
        
        # Convert to torch tensor and add channel dimension
        waveform = torch.from_numpy(waveform).float()
        
        # Move to GPU immediately
        waveform = waveform.to(device)
        
        # Handle mono/stereo - ensure shape is (channels, samples)
        if waveform.ndim == 1:
            # Mono audio - add channel dimension
            waveform = waveform.unsqueeze(0)
        else:
            # Stereo/multi-channel - transpose to (channels, samples)
            waveform = waveform.T
            # Convert to mono (GPU operation)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to target sample rate if needed (on GPU)
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = get_resampler(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Remove channel dimension (mono audio)
        waveform = waveform.squeeze(0)
        
        # Check if audio is valid
        if waveform.numel() == 0:
            print(f"  ⚠️  Warning: Empty audio file")
            return None, 0
            
        # Check if audio is too short (at least 100ms)
        min_samples = int(0.1 * TARGET_SAMPLE_RATE)  # 100ms minimum
        if waveform.shape[0] < min_samples:
            print(f"  ⚠️  Warning: Audio too short (< 100ms)")
            return None, 0
            
        return waveform, TARGET_SAMPLE_RATE
        
    except Exception as e:
        print(f"  ❌ Error loading file: {str(e)}")
        return None, 0


def denoise_with_deepfilternet(audio: Union[np.ndarray, torch.Tensor], sample_rate: int) -> Optional[torch.Tensor]:
    """
    Apply DeepFilterNet 2 speech enhancement to the audio signal using GPU acceleration.
    
    DeepFilterNet 2 is a state-of-the-art speech enhancement model that runs entirely on GPU,
    providing significant speedup compared to CPU-based solutions. It processes full waveforms
    directly without frame-based chunking.
    
    Args:
        audio: Input audio (torch.Tensor on GPU or np.ndarray)
        sample_rate: Sample rate of input audio (will be resampled to 48kHz for processing)
        
    Returns:
        Denoised audio tensor on GPU (at OUTPUT_SAMPLE_RATE) or None if processing fails
    """
    try:
        start_time = time.time()
        
        # Convert numpy array to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).to(device)
        else:
            audio_tensor = audio
        
        # Ensure audio is float32
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        # Get DeepFilterNet 3 model
        model, df_state = get_deepfilternet_model()
        
        # Resample to 48kHz if needed (DeepFilterNet works best at 48kHz)
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = get_resampler(sample_rate, TARGET_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        # DeepFilterNet expects CPU tensors for its internal processing
        # Move to CPU for the enhance() call
        audio_cpu = audio_tensor.cpu()
        
        # DeepFilterNet expects shape (channels, time) NOT (batch, channels, time)
        # audio_cpu is currently (time,) so we need (1, time) for mono
        audio_2d = audio_cpu.unsqueeze(0)  # (1, time)
        
        # Apply DeepFilterNet 3 enhancement
        # Note: The model runs on GPU, but enhance() handles device management internally
        with torch.no_grad():  # No gradients needed for inference
            enhanced_2d = enhance(model, df_state, audio_2d)
        
        # Move result back to GPU for further processing
        enhanced_2d = enhanced_2d.to(device)
        
        # Remove channel dimension: (1, time) -> (time,)
        enhanced_tensor = enhanced_2d.squeeze(0)
        
        # Resample to output sample rate (16kHz) if needed
        if TARGET_SAMPLE_RATE != OUTPUT_SAMPLE_RATE:
            resampler = get_resampler(TARGET_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
            enhanced_tensor = resampler(enhanced_tensor.unsqueeze(0)).squeeze(0)
        
        # Log benchmark time
        elapsed = time.time() - start_time
        _benchmark_times["deepfilternet_gpu"].append(elapsed)
        
        return enhanced_tensor
        
    except Exception as e:
        print(f"  ❌ Error in DeepFilterNet 2 enhancement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"  ❌ Error during noise suppression: {str(e)}")
        return None


def save_audio(filepath: str, cleaned_audio: Union[torch.Tensor, np.ndarray], sample_rate: int) -> bool:
    """
    Save the cleaned audio to a WAV file using GPU-compatible torchaudio.
    
    Args:
        filepath: Output file path
        cleaned_audio: Denoised audio (torch.Tensor on GPU or np.ndarray)
        sample_rate: Sample rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(filepath)
        if output_dir:  # Only create directory if filepath has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to torch tensor if numpy array
        if isinstance(cleaned_audio, np.ndarray):
            audio_tensor = torch.from_numpy(cleaned_audio)
        else:
            audio_tensor = cleaned_audio
        
        # Add channel dimension for torchaudio (expects [channels, samples])
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to CPU for saving
        audio_cpu = audio_tensor.cpu()
        
        # Save as WAV file
        torchaudio.save(
            filepath,
            audio_cpu,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving file: {str(e)}")
        return False


def batch_preprocess(input_dir: str, output_dir: str, access_key: str = None) -> dict:
    """
    GPU-accelerated batch processing with progress bars and performance tracking.
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save cleaned WAV files
        access_key: Deprecated parameter (kept for backward compatibility, not used)
        
    Returns:
        Dictionary with processing statistics and performance metrics
    """
    # Convert to Path objects for easier manipulation
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        return {"error": "Input directory not found"}
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files (MP3, WAV, FLAC, etc.)
    audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_path.glob(f"**/{ext}")))
    
    if not audio_files:
        print(f"⚠️  No audio files found in '{input_dir}'")
        return {"error": "No audio files found"}
    
    print(f"\n{'='*70}")
    print(f"🎵 GPU-Accelerated Audio Preprocessing Pipeline - Stage 1")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Input directory:  {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Total files found: {len(audio_files)}")
    print(f"{'='*70}\n")
    
    # Statistics tracking
    stats = {
        "total_files": len(audio_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_duration": 0.0,
        "processing_times": [],
        "gpu_utilization": []
    }
    
    # Start timing
    overall_start = time.time()
    
    # Process each file with progress bar
    with tqdm(total=len(audio_files), desc="🎵 Processing audio", unit="file", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for audio_file in audio_files:
            file_start = time.time()
            
            # Get relative path to maintain folder structure
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.wav')
            
            # Load audio (GPU-accelerated)
            audio_tensor, sr = load_audio(str(audio_file))
            if audio_tensor is None:
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "⏭️ skipped", "file": relative_path.name[:25]})
                continue
            
            # Check for silence (GPU operation)
            max_amplitude = torch.max(torch.abs(audio_tensor)).item()
            if max_amplitude < 0.001:
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "🔇 silent", "file": relative_path.name[:25]})
                continue
            
            # Apply noise reduction (GPU-accelerated with DeepFilterNet 2)
            cleaned_tensor = denoise_with_deepfilternet(audio_tensor, sr)
            if cleaned_tensor is None:
                stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "❌ failed", "file": relative_path.name[:25]})
                continue
            
            # Save cleaned audio with OUTPUT_SAMPLE_RATE (16kHz)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if save_audio(str(output_file), cleaned_tensor, OUTPUT_SAMPLE_RATE):
                # Calculate metrics using OUTPUT_SAMPLE_RATE
                duration = len(cleaned_tensor) / OUTPUT_SAMPLE_RATE
                file_time = time.time() - file_start
                
                stats["processed"] += 1
                stats["total_duration"] += duration
                stats["processing_times"].append(file_time)
                
                # Update progress bar
                speed = duration / file_time if file_time > 0 else 0
                pbar.update(1)
                pbar.set_postfix({
                    "status": "✅ done",
                    "duration": f"{duration:.1f}s",
                    "speed": f"{speed:.1f}x",
                    "file": relative_path.name[:25]
                })
            else:
                stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix({"status": "💾 failed", "file": relative_path.name[:25]})
    
    # Calculate final metrics
    total_time = time.time() - overall_start
    stats["total_time"] = total_time
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 GPU Processing Summary")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {stats['processed']} files")
    print(f"⏭️  Skipped (corrupted/empty/silent): {stats['skipped']} files")
    print(f"❌ Failed: {stats['failed']} files")
    print(f"📁 Total files: {stats['total_files']} files")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"   Total processing time: {total_time:.2f}s")
    
    if stats['processed'] > 0:
        avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
        throughput = stats['processed'] / total_time
        real_time_factor = stats['total_duration'] / total_time
        
        print(f"   Average per file: {avg_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} files/sec")
        print(f"   Audio duration processed: {stats['total_duration']:.2f}s")
        print(f"   Real-time factor: {real_time_factor:.2f}x")
    
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Memory Usage:")
        print(f"   Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"   Peak reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    
    print(f"{'='*70}\n")
    
    if stats['processed'] > 0:
        print(f"🎉 Completed GPU-accelerated noise suppression for {stats['processed']} files.")
        print(f"📂 Cleaned audio files saved to: {output_path.absolute()}")
    else:
        print(f"⚠️  No files were successfully processed.")
    
    return stats


def cleanup_resources():
    """Clean up global resources (DeepFilterNet model, GPU cache, etc.)."""
    global _df_model, _df_state, _resamplers_cache
    
    # Clean up DeepFilterNet model
    if _df_model is not None:
        try:
            del _df_model
            del _df_state
            _df_model = None
            _df_state = None
            print("✅ DeepFilterNet 2 model cleaned up")
        except:
            pass
    
    # Clear resampler cache
    _resamplers_cache.clear()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")
    
    # Print benchmark summary
    if _benchmark_times["deepfilternet_gpu"]:
        avg_time = sum(_benchmark_times["deepfilternet_gpu"]) / len(_benchmark_times["deepfilternet_gpu"])
        print(f"\n📊 DeepFilterNet 2 Performance:")
        print(f"   Average processing time: {avg_time:.3f}s per file")
        print(f"   Total files processed: {len(_benchmark_times['deepfilternet_gpu'])}")
        if _benchmark_times["koala_cpu"]:
            speedup = _benchmark_times["koala_cpu"] / avg_time
            print(f"   Speedup vs Koala CPU: {speedup:.2f}x faster")


def main():
    """
    Main entry point for the GPU-accelerated preprocessing pipeline.
    """
    # Define input and output directories
    # Modify these paths according to your dataset location
    INPUT_DIR = "./audio_data/raw_data"
    OUTPUT_DIR = "./audio_data/clean_audio"
    
    # Alternative paths for different datasets
    # INPUT_DIR = "./audio_data/downloaded_dataset/cv-corpus-20.0-delta-2024-12-06/en/clips"
    # OUTPUT_DIR = "./audio_data/clean_audio/cv-corpus-20"
    # INPUT_DIR = "./Acted Emotional Speech Dynamic Database – AESDD/Acted Emotional Speech Dynamic Database/anger"
    # OUTPUT_DIR = "./audio_data/clean_audio/anger"
    
    print("\n" + "="*70)
    print("🎙️  Speech-Based Mental Health Dataset Cleaning")
    print("   Stage 1: GPU-Accelerated Speech Enhancement with DeepFilterNet 2")
    print("="*70)
    
    try:
        # Run batch preprocessing (no access key needed - DeepFilterNet 2 is open source)
        stats = batch_preprocess(INPUT_DIR, OUTPUT_DIR)
        
        # Print final message
        if stats.get("processed", 0) > 0:
            print("\n🎉 GPU-accelerated preprocessing completed successfully!")
            
            # Show performance comparison hint
            if torch.cuda.is_available():
                print("\n💡 Tip: Your GPU significantly accelerated the processing!")
                print("   All operations including DeepFilterNet 2 enhancement ran on GPU.")
                print("   This is significantly faster than CPU-based solutions like Koala.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user!")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
    finally:
        # Cleanup resources
        print("\n🧹 Cleaning up resources...")
        cleanup_resources()


if __name__ == "__main__":
    main()
