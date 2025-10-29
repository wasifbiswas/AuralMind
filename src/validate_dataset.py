"""
Data Validation Script
Checks for corrupted or empty .npy files and creates a clean dataset
"""

import os
import numpy as np
from tqdm import tqdm
import shutil

def validate_and_clean_dataset(dataset_dir, output_dir=None):
    """
    Validate all .npy files and optionally copy valid ones to output directory
    
    Args:
        dataset_dir: Directory containing .npy files
        output_dir: Optional directory to copy valid files (if None, removes invalid in-place)
    """
    print(f"\n{'='*80}")
    print(f" VALIDATING DATASET: {os.path.basename(dataset_dir)}")
    print(f"{'='*80}\n")
    
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    print(f"Total files found: {len(all_files)}")
    
    valid_files = []
    invalid_files = []
    
    print("\nValidating files...")
    for filename in tqdm(all_files):
        filepath = os.path.join(dataset_dir, filename)
        
        try:
            # Try to load the file
            data = np.load(filepath)
            
            # Check if it's empty or has wrong shape
            if data.size == 0:
                invalid_files.append((filename, "Empty array"))
                continue
            
            if len(data.shape) != 2:
                invalid_files.append((filename, f"Wrong shape: {data.shape}"))
                continue
            
            if data.shape[1] != 1024:
                invalid_files.append((filename, f"Wrong feature dim: {data.shape[1]}"))
                continue
            
            # Valid file
            valid_files.append(filename)
            
        except Exception as e:
            invalid_files.append((filename, str(e)))
    
    # Print results
    print(f"\n{'='*80}")
    print(" VALIDATION RESULTS")
    print(f"{'='*80}\n")
    print(f"✓ Valid files: {len(valid_files)}")
    print(f"✗ Invalid files: {len(invalid_files)}")
    print(f"  Success rate: {len(valid_files)/len(all_files)*100:.2f}%")
    
    if invalid_files:
        print(f"\n{'='*80}")
        print(" INVALID FILES")
        print(f"{'='*80}\n")
        for filename, reason in invalid_files[:20]:  # Show first 20
            print(f"  {filename}: {reason}")
        
        if len(invalid_files) > 20:
            print(f"\n  ... and {len(invalid_files) - 20} more")
    
    # Handle invalid files
    if invalid_files:
        print(f"\n{'='*80}")
        print(" CLEANING OPTIONS")
        print(f"{'='*80}\n")
        
        if output_dir:
            # Copy valid files to output directory
            os.makedirs(output_dir, exist_ok=True)
            print(f"Copying {len(valid_files)} valid files to: {output_dir}")
            
            for filename in tqdm(valid_files):
                src = os.path.join(dataset_dir, filename)
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)
            
            print(f"✓ Copied {len(valid_files)} valid files")
        else:
            # Remove invalid files from original directory
            response = input(f"\nRemove {len(invalid_files)} invalid files from original directory? (yes/no): ")
            
            if response.lower() == 'yes':
                print("\nRemoving invalid files...")
                for filename, _ in tqdm(invalid_files):
                    filepath = os.path.join(dataset_dir, filename)
                    os.remove(filepath)
                print(f"✓ Removed {len(invalid_files)} invalid files")
            else:
                print("Skipped removal. Invalid files remain in directory.")
    
    print(f"\n{'='*80}")
    print(" SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Ready for training: {'YES' if len(valid_files) > 0 else 'NO'}")
    print(f"{'='*80}\n")
    
    return valid_files, invalid_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and clean dataset')
    parser.add_argument('--dataset', type=str, default='CommonDB',
                       choices=['IEMOCAP', 'CommonDB'],
                       help='Dataset to validate')
    parser.add_argument('--remove-invalid', action='store_true',
                       help='Automatically remove invalid files')
    
    args = parser.parse_args()
    
    dataset_dir = f"d:/MCA Minor1/src/audio_data/features/{args.dataset}"
    
    # Validate and clean
    valid, invalid = validate_and_clean_dataset(dataset_dir)
    
    # Auto-remove if flag set
    if args.remove_invalid and invalid:
        print(f"\nAuto-removing {len(invalid)} invalid files...")
        for filename, _ in invalid:
            filepath = os.path.join(dataset_dir, filename)
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to remove {filename}: {e}")
        print("✓ Cleanup complete!")
