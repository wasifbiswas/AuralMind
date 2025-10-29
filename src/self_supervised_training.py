"""
Self-Supervised Audio Classification Pipeline with Pseudo-Labeling
===================================================================

This script implements a complete pipeline for training on unlabeled audio features:
1. Load preprocessed .npy feature files (variable-length sequences)
2. Apply on-the-fly data augmentation (noise addition, time masking)
3. CNN + BiLSTM model for temporal-spatial feature extraction
4. Pseudo-label generation using K-Means clustering
5. Iterative MLP classifier training until 85% accuracy is achieved
6. GPU-optimized for RTX 4060

Author: AuralMind Team
Date: October 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the training pipeline"""
    
    # Dataset selection (will be set from command line or default)
    DATASET_NAME = "IEMOCAP"  # Default dataset
    
    # Paths (dynamically configured based on DATASET_NAME)
    @classmethod
    def set_dataset(cls, dataset_name):
        """Configure paths for specific dataset"""
        cls.DATASET_NAME = dataset_name
        cls.FEATURES_DIR = f"d:/MCA Minor1/src/audio_data/features/{dataset_name}"
        cls.EMBEDDINGS_DIR = f"d:/MCA Minor1/embeddings_{dataset_name}"
        cls.MODELS_DIR = f"d:/MCA Minor1/models_{dataset_name}"
        
        # Create output directories
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
    
    # Default paths (for IEMOCAP)
    FEATURES_DIR = "d:/MCA Minor1/src/audio_data/features/IEMOCAP"
    EMBEDDINGS_DIR = "d:/MCA Minor1/embeddings_IEMOCAP"
    MODELS_DIR = "d:/MCA Minor1/models_IEMOCAP"
    
    # Data parameters
    MAX_SEQ_LENGTH = 500  # Maximum sequence length for padding/truncation
    FEATURE_DIM = 1024    # wav2vec2-large-xlsr-53 output dimension
    
    # Model parameters
    CNN_CHANNELS = [64, 128, 256]  # CNN channel progression
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    EMBEDDING_DIM = 512   # Final embedding dimension after CNN+BiLSTM
    
    # Clustering parameters
    NUM_CLUSTERS = 5      # Number of pseudo-label clusters (adjust based on your task)
    
    # MLP Classifier parameters
    MLP_HIDDEN_DIMS = [256, 128]
    MLP_DROPOUT = 0.4
    
    # Training parameters
    BATCH_SIZE = 64       # Optimal for RTX 4060 (balance between throughput and memory)
    LEARNING_RATE = 0.0001
    NUM_EPOCHS_EMBEDDING = 15  # Reduced for faster training (still effective)
    NUM_EPOCHS_CLASSIFIER = 15  # Reduced per iteration
    TARGET_ACCURACY = 0.85      # Target accuracy threshold
    MAX_ITERATIONS = 10         # Maximum pseudo-labeling iterations
    
    # Data loading optimization (balanced settings to avoid RAM exhaustion)
    NUM_WORKERS = 4       # Moderate parallel data loading (balance performance vs RAM)
    PREFETCH_FACTOR = 3   # Moderate prefetching (avoid RAM overflow)
    
    # Mixed precision training
    USE_AMP = True        # Automatic Mixed Precision for faster training
    
    # Performance tuning
    CUDNN_BENCHMARK = True  # Enable cuDNN auto-tuner for optimal algorithms
    
    # Training optimization
    GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation for faster updates
    
    # Augmentation parameters
    NOISE_FACTOR = 0.005
    TIME_MASK_PARAM = 20  # Maximum time steps to mask
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)


# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def add_gaussian_noise(features, noise_factor=0.005, device='cuda'):
    """
    Add Gaussian noise to features for augmentation (GPU-optimized)
    
    Args:
        features: Input feature tensor [seq_len, feature_dim]
        noise_factor: Standard deviation of noise
        device: Device to generate noise on (prefer 'cuda' for speed)
    
    Returns:
        Augmented features
    """
    # Generate noise directly on GPU if available for better performance
    if features.is_cuda or device == 'cuda':
        noise = torch.randn_like(features, device=features.device) * noise_factor
    else:
        noise = torch.randn_like(features) * noise_factor
    return features + noise


def time_masking(features, time_mask_param=20):
    """
    Apply time masking augmentation (SpecAugment style) - optimized version
    
    Args:
        features: Input feature tensor [seq_len, feature_dim]
        time_mask_param: Maximum number of time steps to mask
    
    Returns:
        Time-masked features
    """
    seq_len = features.shape[0]
    if seq_len <= time_mask_param:
        return features
    
    # Random mask length and start (vectorized)
    mask_len = np.random.randint(0, time_mask_param)
    mask_start = np.random.randint(0, seq_len - mask_len)
    
    # Apply mask (set to zero) - in-place operation for efficiency
    features[mask_start:mask_start + mask_len, :] = 0
    
    return features


# ============================================================================
# DATASET CLASS
# ============================================================================

class UnlabeledAudioDataset(Dataset):
    """
    Dataset class for loading unlabeled audio features with on-the-fly augmentation
    
    Features are loaded from .npy files and augmented during training.
    Supports variable-length sequences with padding/truncation.
    """
    
    def __init__(self, features_dir, max_seq_length=500, augment=True, 
                 noise_factor=0.005, time_mask_param=20, pseudo_labels=None):
        """
        Args:
            features_dir: Directory containing .npy feature files
            max_seq_length: Maximum sequence length for padding/truncation
            augment: Whether to apply data augmentation
            noise_factor: Noise level for augmentation
            time_mask_param: Time masking parameter
            pseudo_labels: Optional pseudo-labels for supervised training
        """
        self.features_dir = features_dir
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.noise_factor = noise_factor
        self.time_mask_param = time_mask_param
        self.pseudo_labels = pseudo_labels
        
        # Get all .npy files
        self.file_list = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
        print(f"Found {len(self.file_list)} feature files in {features_dir}")
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npy files found in {features_dir}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Load and process a single feature file
        
        Returns:
            features: Padded/truncated feature tensor [max_seq_length, feature_dim]
            seq_length: Original sequence length (before padding)
            label: Pseudo-label if available, else -1
            filename: Original filename for tracking
        """
        filename = self.file_list[idx]
        filepath = os.path.join(self.features_dir, filename)
        
        # Load features from .npy file
        features = np.load(filepath)  # Shape: [seq_len, feature_dim]
        features = torch.FloatTensor(features)
        
        # Store original length
        original_length = features.shape[0]
        
        # Pad or truncate to max_seq_length
        if original_length > self.max_seq_length:
            # Truncate
            features = features[:self.max_seq_length, :]
            seq_length = self.max_seq_length
        else:
            # Pad with zeros
            padding = torch.zeros(self.max_seq_length - original_length, features.shape[1])
            features = torch.cat([features, padding], dim=0)
            seq_length = original_length
        
        # Apply augmentation during training
        if self.augment:
            # Add Gaussian noise
            features = add_gaussian_noise(features, self.noise_factor)
            # Apply time masking (only on non-padded regions)
            if seq_length > 0:
                features[:seq_length, :] = time_masking(
                    features[:seq_length, :], 
                    self.time_mask_param
                )
        
        # Get pseudo-label if available
        label = int(self.pseudo_labels[idx]) if self.pseudo_labels is not None else -1
        
        return features, seq_length, label, filename


# ============================================================================
# CNN + BIDIRECTIONAL LSTM MODEL
# ============================================================================

class CNNBiLSTMEncoder(nn.Module):
    """
    Hybrid CNN + Bidirectional LSTM model for temporal-spatial feature extraction
    
    Architecture:
    1. 1D Convolutional layers for local feature extraction
    2. Bidirectional LSTM for temporal modeling
    3. Global pooling to create fixed-length embeddings
    """
    
    def __init__(self, input_dim, cnn_channels, lstm_hidden_size, 
                 lstm_num_layers, lstm_dropout, embedding_dim):
        """
        Args:
            input_dim: Input feature dimension (1024 for wav2vec2)
            cnn_channels: List of CNN output channels
            lstm_hidden_size: Hidden size for LSTM
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: Dropout rate for LSTM
            embedding_dim: Final embedding dimension
        """
        super(CNNBiLSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # === CNN Layers ===
        # 1D convolutions operate on [batch, channels, seq_len]
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # === Bidirectional LSTM ===
        # Input: [batch, seq_len, features]
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # === Projection to Embedding Space ===
        # BiLSTM outputs: hidden_size * 2 (bidirectional)
        lstm_output_dim = lstm_hidden_size * 2
        
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # === Global Pooling ===
        # We'll use both max and average pooling, then concatenate
        self.pool_projection = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, x, seq_lengths):
        """
        Forward pass
        
        Args:
            x: Input features [batch, seq_len, feature_dim]
            seq_lengths: Original sequence lengths before padding [batch]
        
        Returns:
            embeddings: Fixed-length embedding vectors [batch, embedding_dim]
        """
        batch_size = x.size(0)
        
        # === CNN Stage ===
        # Reshape for CNN: [batch, feature_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.cnn(x)  # Output: [batch, cnn_channels[-1], reduced_seq_len]
        
        # Transpose back for LSTM: [batch, reduced_seq_len, cnn_channels[-1]]
        x = x.transpose(1, 2)
        
        # === BiLSTM Stage ===
        # Pack padded sequence for efficient LSTM processing
        # Adjust seq_lengths for CNN pooling layers (3 pooling layers with stride 2)
        adjusted_lengths = (seq_lengths / (2 ** 3)).long().clamp(min=1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, adjusted_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # lstm_output: [batch, reduced_seq_len, lstm_hidden_size * 2]
        
        # === Projection ===
        # Apply projection to each time step
        seq_len = lstm_output.size(1)
        lstm_output_reshaped = lstm_output.contiguous().view(-1, lstm_output.size(2))
        projected = self.projection(lstm_output_reshaped)
        projected = projected.view(batch_size, seq_len, -1)
        # projected: [batch, reduced_seq_len, embedding_dim]
        
        # === Global Pooling ===
        # Max pooling over time
        max_pool = torch.max(projected, dim=1)[0]  # [batch, embedding_dim]
        # Average pooling over time
        avg_pool = torch.mean(projected, dim=1)    # [batch, embedding_dim]
        
        # Concatenate and project to final embedding dimension
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # [batch, embedding_dim * 2]
        embeddings = self.pool_projection(pooled)  # [batch, embedding_dim]
        
        return embeddings


# ============================================================================
# MLP CLASSIFIER
# ============================================================================

class MLPClassifier(nn.Module):
    """
    Modular MLP classifier that takes embeddings as input
    
    This classifier is trained on pseudo-labeled embeddings from the encoder.
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.4):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, embeddings):
        """
        Forward pass
        
        Args:
            embeddings: Input embeddings [batch, embedding_dim]
        
        Returns:
            logits: Class logits [batch, num_classes]
        """
        return self.classifier(embeddings)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def print_gpu_memory():
    """Print current GPU memory usage and utilization"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Try to get GPU utilization (requires nvidia-ml-py)
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Memory: {allocated:.2f}GB/{total:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
            print(f"GPU Utilization: {util.gpu}% | Memory Utilization: {util.memory}%")
        except:
            print(f"GPU Memory: {allocated:.2f}GB/{total:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")


def train_encoder_unsupervised(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    """
    Train the CNN+BiLSTM encoder in an unsupervised manner
    
    Uses a contrastive or reconstruction loss (here we use a simple autoencoder approach)
    
    Args:
        model: CNNBiLSTMEncoder model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}: Training Encoder (Unsupervised)")
    print(f"{'='*60}")
    
    for batch_idx, (features, seq_lengths, _, _) in enumerate(dataloader):
        features = features.to(device, non_blocking=True)
        seq_lengths = seq_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                embeddings = model(features, seq_lengths)
                
                # For unsupervised learning, we use a simple reconstruction loss
                # The model learns to create meaningful embeddings by reconstructing input statistics
                # Here we use a contrastive approach: embeddings should be consistent
                
                # Simple approach: encourage embeddings to have similar statistics to input
                # This is a placeholder - you can replace with proper contrastive loss
                target_mean = features.mean(dim=(1, 2))
                target_std = features.std(dim=(1, 2))
                
                pred_mean = embeddings.mean(dim=1)
                pred_std = embeddings.std(dim=1)
                
                loss = criterion(pred_mean, target_mean) + criterion(pred_std, target_std)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass (without mixed precision)
            embeddings = model(features, seq_lengths)
            
            target_mean = features.mean(dim=(1, 2))
            target_std = features.std(dim=(1, 2))
            
            pred_mean = embeddings.mean(dim=1)
            pred_std = embeddings.std(dim=1)
            
            loss = criterion(pred_mean, target_mean) + criterion(pred_std, target_std)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            print_gpu_memory()
    
    avg_loss = total_loss / num_batches
    print(f"\nEpoch {epoch} Average Loss: {avg_loss:.4f}")
    
    return avg_loss


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from the encoder for all samples
    
    Args:
        model: Trained CNNBiLSTMEncoder model
        dataloader: DataLoader for data
        device: Device to use
    
    Returns:
        embeddings: NumPy array of embeddings [num_samples, embedding_dim]
        filenames: List of filenames corresponding to embeddings
    """
    model.eval()
    all_embeddings = []
    all_filenames = []
    
    print("\n" + "="*60)
    print("Extracting Embeddings...")
    print("="*60)
    
    with torch.no_grad():
        for batch_idx, (features, seq_lengths, _, filenames) in enumerate(dataloader):
            features = features.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            
            # Extract embeddings
            embeddings = model(features, seq_lengths)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(filenames)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    embeddings = np.vstack(all_embeddings)
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    
    return embeddings, all_filenames


def generate_pseudo_labels(embeddings, num_clusters=5):
    """
    Generate pseudo-labels using K-Means clustering
    
    Args:
        embeddings: NumPy array of embeddings [num_samples, embedding_dim]
        num_clusters: Number of clusters
    
    Returns:
        pseudo_labels: NumPy array of cluster assignments [num_samples]
        silhouette: Silhouette score (quality metric)
    """
    print("\n" + "="*60)
    print(f"Generating Pseudo-Labels with K-Means (k={num_clusters})...")
    print("="*60)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300)
    pseudo_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score (measures cluster quality)
    silhouette = silhouette_score(embeddings, pseudo_labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Cluster distribution: {np.bincount(pseudo_labels)}")
    
    return pseudo_labels, silhouette


def train_classifier(classifier, embeddings, labels, device, num_epochs=20, learning_rate=0.0001):
    """
    Train the MLP classifier on pseudo-labeled embeddings
    
    Args:
        classifier: MLPClassifier model
        embeddings: NumPy array of embeddings [num_samples, embedding_dim]
        labels: NumPy array of labels [num_samples]
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        best_accuracy: Best validation accuracy achieved
        train_accuracy: Final training accuracy
    """
    print("\n" + "="*60)
    print("Training MLP Classifier on Pseudo-Labels...")
    print("="*60)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    # Training setup
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    train_accuracy = 0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        classifier.train()
        
        # Forward pass
        logits = classifier(X_train)
        loss = criterion(logits, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracies
        with torch.no_grad():
            classifier.eval()
            
            # Training accuracy
            train_preds = torch.argmax(classifier(X_train), dim=1)
            train_accuracy = (train_preds == y_train).float().mean().item()
            
            # Validation accuracy
            val_logits = classifier(X_val)
            val_preds = torch.argmax(val_logits, dim=1)
            val_accuracy = (val_preds == y_val).float().mean().item()
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{num_epochs}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
    
    print(f"\nBest Validation Accuracy: {best_accuracy:.4f}")
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    
    return best_accuracy, train_accuracy


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline for self-supervised learning with pseudo-labeling
    
    Pipeline:
    1. Load unlabeled audio features
    2. Train CNN+BiLSTM encoder in unsupervised manner
    3. Extract embeddings from trained encoder
    4. Generate pseudo-labels using K-Means clustering
    5. Train MLP classifier on pseudo-labeled embeddings
    6. Repeat steps 3-5 iteratively until target accuracy is reached
    """
    
    print("="*80)
    print(" Self-Supervised Audio Classification with Pseudo-Labeling")
    print("="*80)
    print(f"\nDevice: {Config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Enable cuDNN benchmarking for optimal performance
        if Config.CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
            print("cuDNN auto-tuner ENABLED for optimal algorithm selection")
        
        print_gpu_memory()
    
    # Create directories
    Config.create_dirs()
    
    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    
    # Initial dataset without pseudo-labels (for encoder training)
    dataset = UnlabeledAudioDataset(
        features_dir=Config.FEATURES_DIR,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        augment=True,
        noise_factor=Config.NOISE_FACTOR,
        time_mask_param=Config.TIME_MASK_PARAM,
        pseudo_labels=None
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
    
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Number of batches: {len(dataloader)}")
    
    # ========================================================================
    # STEP 2: Initialize CNN+BiLSTM Encoder
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Initializing CNN+BiLSTM Encoder")
    print("="*80)
    
    encoder = CNNBiLSTMEncoder(
        input_dim=Config.FEATURE_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=Config.LSTM_NUM_LAYERS,
        lstm_dropout=Config.LSTM_DROPOUT,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    
    print(f"\nEncoder Architecture:")
    print(encoder)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # STEP 3: Train Encoder (Unsupervised)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Training Encoder (Unsupervised)")
    print("="*80)
    
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=Config.LEARNING_RATE)
    criterion_encoder = nn.MSELoss()
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP and torch.cuda.is_available() else None
    if scaler:
        print("Mixed Precision Training (AMP) ENABLED for faster computation")
    
    for epoch in range(1, Config.NUM_EPOCHS_EMBEDDING + 1):
        train_encoder_unsupervised(
            encoder, dataloader, optimizer_encoder, criterion_encoder, 
            Config.DEVICE, epoch, scaler
        )
        
        # Save encoder checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(Config.MODELS_DIR, f"encoder_epoch_{epoch}.pt")
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"Saved encoder checkpoint: {checkpoint_path}")
    
    # Save final encoder
    encoder_path = os.path.join(Config.MODELS_DIR, "encoder_final.pt")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"\nSaved final encoder: {encoder_path}")
    
    # ========================================================================
    # STEP 4: Iterative Pseudo-Labeling and Classifier Training
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Iterative Pseudo-Labeling & Classifier Training")
    print("="*80)
    
    # Dataset for embedding extraction (no augmentation)
    dataset_no_aug = UnlabeledAudioDataset(
        features_dir=Config.FEATURES_DIR,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        augment=False,
        pseudo_labels=None
    )
    
    dataloader_no_aug = DataLoader(
        dataset_no_aug,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    # Initialize classifier
    classifier = MLPClassifier(
        input_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.MLP_HIDDEN_DIMS,
        num_classes=Config.NUM_CLUSTERS,
        dropout=Config.MLP_DROPOUT
    ).to(Config.DEVICE)
    
    print(f"\nClassifier Architecture:")
    print(classifier)
    
    # Iterative training loop
    best_overall_accuracy = 0
    iteration = 0
    
    while iteration < Config.MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'#'*80}")
        print(f"# ITERATION {iteration}/{Config.MAX_ITERATIONS}")
        print(f"{'#'*80}")
        
        # Extract embeddings
        embeddings, filenames = extract_embeddings(encoder, dataloader_no_aug, Config.DEVICE)
        
        # Save embeddings
        embeddings_path = os.path.join(Config.EMBEDDINGS_DIR, f"embeddings_iter_{iteration}.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings: {embeddings_path}")
        
        # Generate pseudo-labels
        pseudo_labels, silhouette = generate_pseudo_labels(embeddings, Config.NUM_CLUSTERS)
        
        # Save pseudo-labels
        labels_path = os.path.join(Config.EMBEDDINGS_DIR, f"pseudo_labels_iter_{iteration}.npy")
        np.save(labels_path, pseudo_labels)
        print(f"Saved pseudo-labels: {labels_path}")
        
        # Train classifier on pseudo-labeled embeddings
        val_accuracy, train_accuracy = train_classifier(
            classifier, embeddings, pseudo_labels, Config.DEVICE,
            num_epochs=Config.NUM_EPOCHS_CLASSIFIER,
            learning_rate=Config.LEARNING_RATE
        )
        
        # Update best accuracy
        if val_accuracy > best_overall_accuracy:
            best_overall_accuracy = val_accuracy
            
            # Save best classifier
            classifier_path = os.path.join(Config.MODELS_DIR, "classifier_best.pt")
            torch.save(classifier.state_dict(), classifier_path)
            print(f"\n*** New best classifier saved: {classifier_path} ***")
        
        # Check if target accuracy is reached
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} Summary:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Best Overall Accuracy: {best_overall_accuracy:.4f}")
        print(f"  Target Accuracy: {Config.TARGET_ACCURACY:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"{'='*80}")
        
        if val_accuracy >= Config.TARGET_ACCURACY:
            print(f"\n{'*'*80}")
            print(f"*** TARGET ACCURACY REACHED! ***")
            print(f"*** Final Validation Accuracy: {val_accuracy:.4f} ***")
            print(f"{'*'*80}")
            break
        
        # Fine-tune encoder based on pseudo-labels (optional)
        # This step can help improve embeddings in subsequent iterations
        print("\nFine-tuning encoder with pseudo-labels...")
        
        # Update dataset with pseudo-labels
        dataset_labeled = UnlabeledAudioDataset(
            features_dir=Config.FEATURES_DIR,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            augment=True,
            noise_factor=Config.NOISE_FACTOR,
            time_mask_param=Config.TIME_MASK_PARAM,
            pseudo_labels=pseudo_labels
        )
        
        # Use fewer workers during fine-tuning to avoid memory issues
        dataloader_labeled = DataLoader(
            dataset_labeled,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # Reduced to avoid RAM exhaustion
            pin_memory=True,
            prefetch_factor=2,  # Reduced prefetching
            persistent_workers=False  # Don't keep workers persistent to save memory
        )
        
        # Fine-tune encoder for a few epochs
        optimizer_finetune = optim.Adam(encoder.parameters(), lr=Config.LEARNING_RATE * 0.1)
        criterion_finetune = nn.CrossEntropyLoss()
        
        # Create temporary classifier head for encoder fine-tuning
        temp_classifier = nn.Linear(Config.EMBEDDING_DIM, Config.NUM_CLUSTERS).to(Config.DEVICE)
        optimizer_temp = optim.Adam(
            list(encoder.parameters()) + list(temp_classifier.parameters()),
            lr=Config.LEARNING_RATE * 0.1
        )
        
        encoder.train()
        for epoch in range(1, 6):  # Fine-tune for 5 epochs
            total_loss = 0
            for features, seq_lengths, labels, _ in dataloader_labeled:
                features = features.to(Config.DEVICE, non_blocking=True)
                seq_lengths = seq_lengths.to(Config.DEVICE, non_blocking=True)
                labels = labels.long().to(Config.DEVICE, non_blocking=True)  # Convert to Long tensor
                
                optimizer_temp.zero_grad(set_to_none=True)
                
                # Forward pass
                embeddings = encoder(features, seq_lengths)
                logits = temp_classifier(embeddings)
                loss = criterion_finetune(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer_temp.step()
                
                total_loss += loss.item()
            
            print(f"  Fine-tune Epoch {epoch}/5: Loss={total_loss/len(dataloader_labeled):.4f}")
        
        # Save fine-tuned encoder
        encoder_path = os.path.join(Config.MODELS_DIR, f"encoder_iter_{iteration}.pt")
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved fine-tuned encoder: {encoder_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Total Iterations: {iteration}")
    print(f"  Best Validation Accuracy: {best_overall_accuracy:.4f}")
    print(f"  Target Accuracy: {Config.TARGET_ACCURACY:.4f}")
    print(f"  Status: {'ACHIEVED' if best_overall_accuracy >= Config.TARGET_ACCURACY else 'NOT ACHIEVED'}")
    
    print(f"\nSaved Models:")
    print(f"  Encoder: {os.path.join(Config.MODELS_DIR, 'encoder_final.pt')}")
    print(f"  Classifier: {os.path.join(Config.MODELS_DIR, 'classifier_best.pt')}")
    
    print(f"\nSaved Embeddings:")
    print(f"  Directory: {Config.EMBEDDINGS_DIR}")
    
    if torch.cuda.is_available():
        print("\nFinal GPU Memory:")
        print_gpu_memory()
    
    print("\n" + "="*80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Self-Supervised Audio Feature Classification')
    parser.add_argument('--dataset', type=str, default='IEMOCAP', 
                       choices=['IEMOCAP', 'CommonDB'],
                       help='Dataset to use for training (default: IEMOCAP)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of encoder training epochs (default: 15)')
    parser.add_argument('--clusters', type=int, default=None,
                       help='Number of clusters for pseudo-labeling (default: 5)')
    parser.add_argument('--target-acc', type=float, default=None,
                       help='Target accuracy to reach (default: 0.85)')
    
    args = parser.parse_args()
    
    # Configure dataset
    print(f"\n{'='*80}")
    print(f" CONFIGURING FOR {args.dataset} DATASET")
    print(f"{'='*80}\n")
    Config.set_dataset(args.dataset)
    
    # Override config if arguments provided
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        print(f"✓ Batch size set to: {args.batch_size}")
    
    if args.epochs is not None:
        Config.ENCODER_PRETRAIN_EPOCHS = args.epochs
        print(f"✓ Encoder epochs set to: {args.epochs}")
    
    if args.clusters is not None:
        Config.NUM_CLUSTERS = args.clusters
        print(f"✓ Number of clusters set to: {args.clusters}")
    
    if args.target_acc is not None:
        Config.TARGET_ACCURACY = args.target_acc
        print(f"✓ Target accuracy set to: {args.target_acc}")
    
    print(f"\nDataset: {Config.DATASET_NAME}")
    print(f"Features: {Config.FEATURES_DIR}")
    print(f"Models Output: {Config.MODELS_DIR}")
    print(f"Embeddings Output: {Config.EMBEDDINGS_DIR}")
    
    # Run training
    main()
