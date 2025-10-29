"""
GPU Utilization Benchmark Script
Tests the optimized training pipeline for GPU utilization
"""

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import time
import numpy as np
from self_supervised_training import Config, UnlabeledAudioDataset, CNNBiLSTMEncoder
from torch.utils.data import DataLoader

def monitor_gpu():
    """Monitor GPU utilization in real-time"""
    if torch.cuda.is_available():
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            
            print(f"\n{'='*60}")
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")
            print(f"Memory Used: {mem_info.used / 1024**3:.2f}GB / {mem_info.total / 1024**3:.2f}GB")
            print(f"{'='*60}")
            
            return util.gpu, util.memory
        except Exception as e:
            print(f"Error monitoring GPU: {e}")
            return 0, 0
    return 0, 0

def benchmark_dataloader():
    """Benchmark data loading speed"""
    print("\n" + "="*80)
    print("BENCHMARKING DATA LOADING")
    print("="*80)
    
    dataset = UnlabeledAudioDataset(
        features_dir=Config.FEATURES_DIR,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        augment=True,
        noise_factor=Config.NOISE_FACTOR,
        time_mask_param=Config.TIME_MASK_PARAM
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Num workers: {Config.NUM_WORKERS}")
    print(f"Total batches: {len(dataloader)}")
    
    start_time = time.time()
    for i, (features, seq_lengths, _, _) in enumerate(dataloader):
        if i >= 50:  # Test first 50 batches
            break
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * Config.BATCH_SIZE / elapsed
            print(f"Batch {i+1}/50: {samples_per_sec:.1f} samples/sec")
    
    total_time = time.time() - start_time
    avg_speed = 50 * Config.BATCH_SIZE / total_time
    print(f"\nAverage speed: {avg_speed:.1f} samples/sec")
    print(f"Time per batch: {total_time/50:.3f}s")

def benchmark_model():
    """Benchmark model training speed and GPU utilization"""
    print("\n" + "="*80)
    print("BENCHMARKING MODEL TRAINING")
    print("="*80)
    
    # Initialize model
    encoder = CNNBiLSTMEncoder(
        input_dim=Config.FEATURE_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=Config.LSTM_NUM_LAYERS,
        lstm_dropout=Config.LSTM_DROPOUT,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None
    
    # Load data
    dataset = UnlabeledAudioDataset(
        features_dir=Config.FEATURES_DIR,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        augment=True,
        noise_factor=Config.NOISE_FACTOR,
        time_mask_param=Config.TIME_MASK_PARAM
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    print(f"Mixed Precision: {Config.USE_AMP}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    
    encoder.train()
    gpu_utils = []
    mem_utils = []
    
    start_time = time.time()
    for i, (features, seq_lengths, _, _) in enumerate(dataloader):
        if i >= 50:  # Test first 50 batches
            break
        
        features = features.to(Config.DEVICE, non_blocking=True)
        seq_lengths = seq_lengths.to(Config.DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
            with torch.cuda.amp.autocast():
                embeddings = encoder(features, seq_lengths)
                target_mean = features.mean(dim=(1, 2))
                target_std = features.std(dim=(1, 2))
                pred_mean = embeddings.mean(dim=1)
                pred_std = embeddings.std(dim=1)
                loss = criterion(pred_mean, target_mean) + criterion(pred_std, target_std)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = encoder(features, seq_lengths)
            target_mean = features.mean(dim=(1, 2))
            target_std = features.std(dim=(1, 2))
            pred_mean = embeddings.mean(dim=1)
            pred_std = embeddings.std(dim=1)
            loss = criterion(pred_mean, target_mean) + criterion(pred_std, target_std)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Monitor GPU every 10 batches
        if (i + 1) % 10 == 0:
            gpu_util, mem_util = monitor_gpu()
            gpu_utils.append(gpu_util)
            mem_utils.append(mem_util)
            
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * Config.BATCH_SIZE / elapsed
            print(f"Batch {i+1}/50: {samples_per_sec:.1f} samples/sec, Loss: {loss.item():.4f}")
    
    total_time = time.time() - start_time
    avg_speed = 50 * Config.BATCH_SIZE / total_time
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Average training speed: {avg_speed:.1f} samples/sec")
    print(f"Time per batch: {total_time/50:.3f}s")
    print(f"Average GPU utilization: {np.mean(gpu_utils):.1f}%")
    print(f"Average memory utilization: {np.mean(mem_utils):.1f}%")
    print(f"Peak GPU utilization: {np.max(gpu_utils):.1f}%")
    print(f"Peak memory utilization: {np.max(mem_utils):.1f}%")
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\nFinal GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print(f"{'='*80}")

if __name__ == "__main__":
    print("="*80)
    print(" GPU UTILIZATION BENCHMARK")
    print("="*80)
    print(f"\nDevice: {Config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Run benchmarks
    monitor_gpu()
    benchmark_dataloader()
    benchmark_model()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
