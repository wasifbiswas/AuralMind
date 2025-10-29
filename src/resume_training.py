"""
Resume Training Script
Continues from saved encoder and iteration 1 results
"""

import os
import numpy as np
import torch
import torch.nn as optim
from torch.utils.data import DataLoader
from self_supervised_training import (
    Config, CNNBiLSTMEncoder, MLPClassifier, UnlabeledAudioDataset,
    extract_embeddings, generate_pseudo_labels, train_classifier, print_gpu_memory
)

def resume_training():
    """Resume training from saved checkpoint"""
    
    print("="*80)
    print(" RESUMING SELF-SUPERVISED TRAINING")
    print("="*80)
    
    # Check if encoder exists
    encoder_path = os.path.join(Config.MODELS_DIR, "encoder_final.pt")
    if not os.path.exists(encoder_path):
        print("ERROR: No saved encoder found. Please run full training first.")
        return
    
    print(f"\nDevice: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        print_gpu_memory()
    
    # Load encoder
    print("\nLoading saved encoder...")
    encoder = CNNBiLSTMEncoder(
        input_dim=Config.FEATURE_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=Config.LSTM_NUM_LAYERS,
        lstm_dropout=Config.LSTM_DROPOUT,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    encoder.load_state_dict(torch.load(encoder_path))
    print("✓ Encoder loaded successfully")
    
    # Load classifier
    classifier_path = os.path.join(Config.MODELS_DIR, "classifier_best.pt")
    classifier = MLPClassifier(
        input_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.MLP_HIDDEN_DIMS,
        num_classes=Config.NUM_CLUSTERS,
        dropout=Config.MLP_DROPOUT
    ).to(Config.DEVICE)
    
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path))
        print("✓ Classifier loaded successfully")
    
    # Check existing iteration
    iteration_files = [f for f in os.listdir(Config.EMBEDDINGS_DIR) if f.startswith("embeddings_iter_")]
    if iteration_files:
        iterations = [int(f.split("_")[-1].split(".")[0]) for f in iteration_files]
        start_iteration = max(iterations) + 1
        print(f"\n✓ Found {max(iterations)} completed iteration(s)")
        print(f"Starting from iteration {start_iteration}")
    else:
        start_iteration = 1
        print("\nStarting from iteration 1")
    
    # Dataset for embedding extraction
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
        num_workers=2,  # Reduced to avoid memory issues
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # Don't persist to save memory
    )
    
    # Continue iterative training
    best_overall_accuracy = 0.4598  # From iteration 1
    
    for iteration in range(start_iteration, Config.MAX_ITERATIONS + 1):
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
        
        # Train classifier
        val_accuracy, train_accuracy = train_classifier(
            classifier, embeddings, pseudo_labels, Config.DEVICE,
            num_epochs=Config.NUM_EPOCHS_CLASSIFIER,
            learning_rate=Config.LEARNING_RATE
        )
        
        # Update best accuracy
        if val_accuracy > best_overall_accuracy:
            best_overall_accuracy = val_accuracy
            classifier_path = os.path.join(Config.MODELS_DIR, "classifier_best.pt")
            torch.save(classifier.state_dict(), classifier_path)
            print(f"\n*** New best classifier saved: {classifier_path} ***")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} Summary:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Best Overall Accuracy: {best_overall_accuracy:.4f}")
        print(f"  Target Accuracy: {Config.TARGET_ACCURACY:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"{'='*80}")
        
        # Check target
        if val_accuracy >= Config.TARGET_ACCURACY:
            print(f"\n{'*'*80}")
            print(f"*** TARGET ACCURACY REACHED! ***")
            print(f"*** Final Validation Accuracy: {val_accuracy:.4f} ***")
            print(f"{'*'*80}")
            break
        
        # Fine-tune encoder (with reduced workers to avoid memory issues)
        print("\nFine-tuning encoder with pseudo-labels...")
        
        dataset_labeled = UnlabeledAudioDataset(
            features_dir=Config.FEATURES_DIR,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            augment=True,
            noise_factor=Config.NOISE_FACTOR,
            time_mask_param=Config.TIME_MASK_PARAM,
            pseudo_labels=pseudo_labels
        )
        
        dataloader_labeled = DataLoader(
            dataset_labeled,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # Reduced to avoid memory issues
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )
        
        optimizer_finetune = torch.optim.Adam(encoder.parameters(), lr=Config.LEARNING_RATE * 0.1)
        criterion_finetune = torch.nn.CrossEntropyLoss()
        
        temp_classifier = torch.nn.Linear(Config.EMBEDDING_DIM, Config.NUM_CLUSTERS).to(Config.DEVICE)
        optimizer_temp = torch.optim.Adam(
            list(encoder.parameters()) + list(temp_classifier.parameters()),
            lr=Config.LEARNING_RATE * 0.1
        )
        
        encoder.train()
        for epoch in range(1, 6):
            total_loss = 0
            for features, seq_lengths, labels, _ in dataloader_labeled:
                features = features.to(Config.DEVICE, non_blocking=True)
                seq_lengths = seq_lengths.to(Config.DEVICE, non_blocking=True)
                labels = labels.long().to(Config.DEVICE, non_blocking=True)  # Convert to Long tensor
                
                optimizer_temp.zero_grad(set_to_none=True)
                
                embeddings = encoder(features, seq_lengths)
                logits = temp_classifier(embeddings)
                loss = criterion_finetune(logits, labels)
                
                loss.backward()
                optimizer_temp.step()
                
                total_loss += loss.item()
            
            print(f"  Fine-tune Epoch {epoch}/5: Loss={total_loss/len(dataloader_labeled):.4f}")
        
        # Save fine-tuned encoder
        encoder_path = os.path.join(Config.MODELS_DIR, f"encoder_iter_{iteration}.pt")
        torch.save(encoder.state_dict(), encoder_path)
        print(f"Saved fine-tuned encoder: {encoder_path}")
    
    # Final summary
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Total Iterations: {iteration}")
    print(f"  Best Validation Accuracy: {best_overall_accuracy:.4f}")
    print(f"  Target Accuracy: {Config.TARGET_ACCURACY:.4f}")
    print(f"  Status: {'ACHIEVED' if best_overall_accuracy >= Config.TARGET_ACCURACY else 'NOT ACHIEVED'}")

if __name__ == "__main__":
    resume_training()
