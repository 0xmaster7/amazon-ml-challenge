import os
import glob
import argparse
import shutil

def get_latest_checkpoint(checkpoint_dir):
    """Finds the checkpoint with the highest epoch/score to resume from."""
    if not os.path.exists(checkpoint_dir):
        return None
    # Look for standard checkpoints, not the best ones
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the extracted dataset")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save/load checkpoints (Google Drive)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    print(f"Data Directory: {args.data_dir}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")

    # Ensure checkpoint directories exist
    best_checkpoint_dir = os.path.join(args.checkpoint_dir, "best_checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    # 1. Check for existing checkpoints to RESUME
    latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
    start_epoch = 0
    global_best_score = 0.0
    best_ckpt_counter = len(glob.glob(os.path.join(best_checkpoint_dir, "best_checkpoint_*.pth")))

    if latest_ckpt:
        print(f"Found existing checkpoint: {latest_ckpt}")
        print("Loading weights and resuming training...")
        # TODO: Load weights and read the previous best score from checkpoint state
    else:
        print("No checkpoints found. Starting fresh training!")

    # 2. Mock Training Loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        print("Training...")
        
        print("Validating...")
        # Mock score that improves initially but maybe fluctuates
        mock_val_score = 0.80 + (epoch * 0.02) 

        # 3. Save standard Checkpoint (for resuming if crashed)
        checkpoint_name = f"model_epoch_{epoch:02d}_score_{mock_val_score:.4f}.pth"
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        
        print(f"Saving standard checkpoint to {checkpoint_path}")
        # Simulate creating a file
        with open(checkpoint_path, 'w') as f:
            f.write("mock_model_weights")

        # 4. Check if it's a NEW BEST score
        if mock_val_score > global_best_score:
            print(f"🎉 New best score achieved! ({mock_val_score:.4f} > {global_best_score:.4f})")
            global_best_score = mock_val_score
            best_ckpt_counter += 1
            
            # Save a copy into the best_checkpoints folder
            best_ckpt_name = f"best_checkpoint_{best_ckpt_counter}_epoch_{epoch:02d}_score_{mock_val_score:.4f}.pth"
            best_ckpt_path = os.path.join(best_checkpoint_dir, best_ckpt_name)
            
            print(f"⭐ Copying to Best Checkpoints folder: {best_ckpt_name}")
            shutil.copy(checkpoint_path, best_ckpt_path)

if __name__ == "__main__":
    main()
