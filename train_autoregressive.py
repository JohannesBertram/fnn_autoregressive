import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from fnn.microns.build import frame_autoregressive_model
import os
from crop_input import segment_and_crop_mp4
import glob

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_steps = 5
num_epochs = 20
batch_size = 8
learning_rate = 1e-4
decay = 0.9
clip_grad_norm_value = 1.0
checkpoint_dir = "checkpoints"

# Ensure checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

mp4_files = glob.glob("input_data/*.mp4")

all_clips = []
for mp4_path in mp4_files:
    print(f"Processing {mp4_path}...")
    clips = segment_and_crop_mp4(
        mp4_path,
        resize=(256, 144),
        diff_threshold=0.49,
        min_clip_length=45,
        crop_length=45,
        save_path=None  # Don't save individual files
    )
    all_clips.append(clips)

seq = np.concatenate(all_clips, axis=0)
seq = torch.tensor(seq).unsqueeze(2).float() / 255.0

full_dataset = TensorDataset(seq)

# Define split ratios (e.g., 80% train, 10% validation, 10% test)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42) # Set seed for reproducibility
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Note: Test loader is kept for completeness, but the full evaluation is typically done separately.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

# --- Model, Loss, and Optimizer Setup ---
# Initialize model with the correct pred_steps
model = frame_autoregressive_model(pred_steps=pred_steps).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Pre-calculate time weights
time_weights = torch.tensor([decay ** t for t in range(pred_steps)], device=device)
time_weights_normalized = time_weights / time_weights.sum()


# --- Training Loop ---
print("\nStarting Training...")
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    
    for step, (batch,) in enumerate(train_loader, start=1):
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        
        # Ensure the clip is long enough for at least 1 input frame + pred_steps
        min_clip_len = pred_steps + 1
        assert T >= min_clip_len, f"Clip length T={T} is too short. Must be at least {min_clip_len}"

        # Randomly choose number of input frames 
        # (at least 1, at most T - pred_steps)
        max_input = T - pred_steps
        # NOTE: np.random.randint(low, high) is [low, high-1]. We need [1, max_input]
        # Assuming the original code meant 'at least 1' input, but used 5 as lower bound.
        # Sticking to the original lower bound of 5 if possible, otherwise use 1.
        lower_bound_input = 5 if max_input >= 5 else 1 
        input_len = np.random.randint(lower_bound_input, max_input + 1)
        
        context = batch[:, :input_len]                              # [B, input_len, 1, H, W]
        target = batch[:, input_len:input_len + pred_steps]         # [B, pred_steps, 1, H, W]

        # Forward pass
        optimizer.zero_grad()
        preds = model(context)  # [B, pred_steps, 1, H, W]

        # Loss calculation (using pre-calculated and normalized time_weights)
        per_frame_loss = torch.stack(
            [criterion(preds[:, t], target[:, t]) for t in range(pred_steps)]
        )  # [pred_steps]
        weighted_loss = (per_frame_loss * time_weights_normalized).sum()
        loss = weighted_loss

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for step, (batch,) in enumerate(val_loader, start=1):
            batch = batch.to(device)
            B, T, C, H, W = batch.shape
            
            min_clip_len = pred_steps + 1
            if T < min_clip_len:
                print(f"Skipping validation batch: Clip length T={T} < {min_clip_len}")
                continue

            # For validation, we can use a fixed, representative input length
            # or continue using random, but fixed for simplicity here.
            # Using the maximum possible input length for a consistent validation measure.
            input_len = T - pred_steps
            
            context = batch[:, :input_len]
            target = batch[:, input_len:input_len + pred_steps]

            preds = model(context)

            # Loss calculation (using pre-calculated and normalized time_weights)
            per_frame_loss = torch.stack(
                [criterion(preds[:, t], target[:, t]) for t in range(pred_steps)]
            )
            weighted_loss = (per_frame_loss * time_weights_normalized).sum()
            total_val_loss += weighted_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # --- Reporting and Checkpointing ---
    print(f"\n--- Epoch {epoch+1}/{num_epochs} Summary ---")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print("-----------------------------------\n")

    # Save checkpoint at end of epoch
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"autoregressive_fnn_epoch_{epoch+1}.pt"))

print("Training complete.")