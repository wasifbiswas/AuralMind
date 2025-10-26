"""
Speech-Based Mental Health Dataset Cleaning (Stage 1)
Preprocessing script for noise reduction using Picovoice Koala SDK

This module handles:
- Loading MP3 audio files
- Converting to 16 kHz mono format
- Applying Koala noise suppression
- Exporting cleaned WAV files
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pvkoala
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Configuration
KOALA_ACCESS_KEY = "4zmm1IWYDar8D7iEJQlwMaEh2wKvIbjYkqyUdxG0OhgvYeWx0mpzjw=="  # actual key
TARGET_SAMPLE_RATE = 16000  # Koala requires 16 kHz
KOALA_FRAME_LENGTH = 256  # Koala processes audio in chunks of 256 samples (pvkoala.Koala.frame_length)


def load_audio(filepath: str) -> Tuple[Optional[np.ndarray], int]:
    """
    Load an MP3 audio file and convert it to mono 16 kHz waveform.
    
    Args:
        filepath: Path to the MP3 file
        
    Returns:
        Tuple of (audio_array, sample_rate) or (None, 0) if loading fails
    """
    try:
        # Load audio file with librosa (auto-converts to mono)
        audio, sr = librosa.load(filepath, sr=TARGET_SAMPLE_RATE, mono=True)
        
        # Check if audio is valid
        if audio is None or len(audio) == 0:
            print(f"  ⚠️  Warning: Empty audio file")
            return None, 0
            
        # Check if audio is too short
        if len(audio) < KOALA_FRAME_LENGTH:
            print(f"  ⚠️  Warning: Audio too short (< {KOALA_FRAME_LENGTH} samples)")
            return None, 0
            
        return audio, sr
        
    except Exception as e:
        print(f"  ❌ Error loading file: {str(e)}")
        return None, 0


def denoise_with_koala(audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """
    Apply Koala noise suppression to the audio signal.
    
    Args:
        audio: Input audio array (mono, 16 kHz)
        sample_rate: Sample rate (must be 16000)
        
    Returns:
        Denoised audio array or None if processing fails
    """
    try:
        # Verify sample rate
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Sample rate must be {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz")
        
        # Initialize Koala
        koala = pvkoala.create(access_key=KOALA_ACCESS_KEY)
        
        # Convert float audio to int16 (Koala expects int16 format)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Process audio in frames
        enhanced_frames = []
        num_frames = len(audio_int16) // KOALA_FRAME_LENGTH
        
        for i in range(num_frames):
            start_idx = i * KOALA_FRAME_LENGTH
            end_idx = start_idx + KOALA_FRAME_LENGTH
            frame = audio_int16[start_idx:end_idx]
            
            # Process frame with Koala
            enhanced_frame = koala.process(frame)
            enhanced_frames.append(enhanced_frame)
        
        # Handle remaining samples (if any)
        remaining_samples = len(audio_int16) % KOALA_FRAME_LENGTH
        if remaining_samples > 0:
            # Pad the last frame to KOALA_FRAME_LENGTH
            last_frame = audio_int16[-remaining_samples:]
            padded_frame = np.pad(last_frame, (0, KOALA_FRAME_LENGTH - remaining_samples), 
                                 mode='constant', constant_values=0)
            enhanced_frame = koala.process(padded_frame)
            # Only keep the valid samples (remove padding)
            enhanced_frames.append(enhanced_frame[:remaining_samples])
        
        # Clean up Koala instance
        koala.delete()
        
        # Concatenate all enhanced frames
        enhanced_audio = np.concatenate(enhanced_frames)
        
        # Convert back to float format (-1.0 to 1.0)
        enhanced_audio_float = enhanced_audio.astype(np.float32) / 32767.0
        
        return enhanced_audio_float
        
    except pvkoala.KoalaActivationError:
        print(f"  ❌ Error: Invalid Koala access key or activation failed")
        return None
    except pvkoala.KoalaActivationLimitError:
        print(f"  ❌ Error: Koala activation limit reached")
        return None
    except Exception as e:
        print(f"  ❌ Error during noise suppression: {str(e)}")
        return None


def save_audio(filepath: str, cleaned_audio: np.ndarray, sample_rate: int) -> bool:
    """
    Save the cleaned audio to a WAV file.
    
    Args:
        filepath: Output file path
        cleaned_audio: Denoised audio array
        sample_rate: Sample rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as WAV file
        sf.write(filepath, cleaned_audio, sample_rate, subtype='PCM_16')
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving file: {str(e)}")
        return False


def batch_preprocess(input_dir: str, output_dir: str, access_key: str = None) -> None:
    """
    Process all MP3 files in the input directory and save cleaned versions.
    
    Args:
        input_dir: Directory containing raw MP3 files
        output_dir: Directory to save cleaned WAV files
        access_key: Picovoice access key (optional, uses global if not provided)
    """
    global KOALA_ACCESS_KEY
    
    # Update access key if provided
    if access_key:
        KOALA_ACCESS_KEY = access_key
    
    # Verify access key is set
    if not KOALA_ACCESS_KEY or KOALA_ACCESS_KEY == "YOUR_PICOVOICE_ACCESS_KEY_HERE":
        print("❌ ERROR: Please set a valid Picovoice access key!")
        print("   Get your free key at: https://console.picovoice.ai/")
        return
    
    # Convert to Path objects for easier manipulation
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all MP3 files
    mp3_files = list(input_path.glob("**/*.mp3"))
    
    if not mp3_files:
        print(f"⚠️  No MP3 files found in '{input_dir}'")
        return
    
    print(f"\n{'='*60}")
    print(f"🎵 Audio Preprocessing Pipeline - Stage 1")
    print(f"{'='*60}")
    print(f"Input directory:  {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Total files found: {len(mp3_files)}")
    print(f"{'='*60}\n")
    
    # Counters for statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process each file
    for idx, mp3_file in enumerate(mp3_files, 1):
        # Get relative path to maintain folder structure
        relative_path = mp3_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.wav')
        
        print(f"[{idx}/{len(mp3_files)}] Processing: {relative_path}")
        
        # Load audio
        audio, sr = load_audio(str(mp3_file))
        if audio is None:
            print(f"  ⏭️  Skipped (corrupted or empty)\n")
            skipped_count += 1
            continue
        
        # Check for silence (audio with very low amplitude)
        if np.max(np.abs(audio)) < 0.001:
            print(f"  ⏭️  Skipped (silent audio)\n")
            skipped_count += 1
            continue
        
        # Apply noise reduction
        print(f"  🔧 Applying Koala noise suppression...")
        cleaned_audio = denoise_with_koala(audio, sr)
        if cleaned_audio is None:
            print(f"  ❌ Failed (noise suppression error)\n")
            failed_count += 1
            continue
        
        # Save cleaned audio
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if save_audio(str(output_file), cleaned_audio, sr):
            print(f"  ✅ Saved cleaned file: {output_file.relative_to(output_path)}")
            print(f"     Duration: {len(cleaned_audio)/sr:.2f}s\n")
            processed_count += 1
        else:
            print(f"  ❌ Failed (save error)\n")
            failed_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Processing Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed_count} files")
    print(f"⏭️  Skipped (corrupted/empty): {skipped_count} files")
    print(f"❌ Failed: {failed_count} files")
    print(f"📁 Total files processed: {len(mp3_files)} files")
    print(f"{'='*60}\n")
    
    if processed_count > 0:
        print(f"🎉 Completed noise suppression for {processed_count} files.")
        print(f"📂 Cleaned audio files saved to: {output_path.absolute()}")
    else:
        print(f"⚠️  No files were successfully processed.")


def main():
    """
    Main entry point for the preprocessing pipeline.
    """
    # Define input and output directories
    # Modify these paths according to your dataset location
    INPUT_DIR = "./audio_data/raw_data"
    OUTPUT_DIR = "./audio_data/clean_audio"
    
    # Alternative: You can also use the downloaded dataset
    # INPUT_DIR = "./audio_data/downloaded_dataset/cv-corpus-20.0-delta-2024-12-06/en/clips"
    # OUTPUT_DIR = "./audio_data/clean_audio/cv-corpus-20"
    
    print("\n" + "="*60)
    print("🎙️  Speech-Based Mental Health Dataset Cleaning")
    print("   Stage 1: Noise Reduction with Picovoice Koala SDK")
    print("="*60)
    
    # Prompt user for access key if not set
    if KOALA_ACCESS_KEY == "YOUR_PICOVOICE_ACCESS_KEY_HERE":
        print("\n⚠️  Picovoice Access Key not configured!")
        print("   Please get your free key at: https://console.picovoice.ai/")
        access_key = input("\nEnter your Picovoice Access Key (or press Enter to use default): ").strip()
        if access_key:
            batch_preprocess(INPUT_DIR, OUTPUT_DIR, access_key)
        else:
            print("\n❌ Cannot proceed without a valid access key.")
    else:
        # Run batch preprocessing
        batch_preprocess(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
