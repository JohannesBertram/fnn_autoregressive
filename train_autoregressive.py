import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from fnn.microns.build import frame_autoregressive_model
import os
from crop_input import segment_and_crop_mp4
import glob
from utils import createFlowDataset, subps 
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_steps = 5
num_epochs = 10
batch_size = 8
learning_rate = 1e-4
decay = 0.9
clip_grad_norm_value = 1.0
checkpoint_dir = "checkpoints"

# Ensure checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

mp4_files = glob.glob("input_data/*.mp4")
#mp4_files = [mp4_files[0]]

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
print(seq.shape)

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

def evaluate(model, epoch=0):

    ############# LOAD FLOW STIM FRAMES #################
    counter = 0
    orig_shape = (800, 600)
    scl_factor = 0.7
    N_INSTANCES = 3
    trial_len = 75 // 2  # Number of frames
    stride = 1
    input_shape = (144, 256)

    mydirs = list(map(str, range(0, 360, 45)))
    categories = ['grat_W12', 'grat_W1', 'grat_W2',
                'neg1dotflow_D1_bg', 'neg3dotflow_D1_bg', 'neg1dotflow_D2_bg', 'neg3dotflow_D2_bg',
                'pos1dotflow_D1_bg', 'pos3dotflow_D1_bg', 'pos1dotflow_D2_bg', 'pos3dotflow_D2_bg']

    topdir = 'flowstims'
    NDIRS = len(mydirs)
    tot_stims = len(categories) * NDIRS
    print('tot_stims', tot_stims, flush=True)
    frames_per_stim = (trial_len // stride)
    print('frames_per_stim', frames_per_stim)

    # Create flow datasets (placeholder function)
    flow_dataset = createFlowDataset(categories, topdir, mydirs, orig_shape, input_shape,
                                    scl_factor, N_INSTANCES, trial_len, stride)[0] # inst 0

    flow_sequences = flow_dataset.reshape((-1, 37, 1, 144, 256)).astype(np.uint8)
    flow_sequences_torch = torch.Tensor(flow_sequences).float() / 255.0
    print(flow_sequences_torch.shape)

    flow_dataset_torch = TensorDataset(flow_sequences_torch)

    flow_loader = DataLoader(flow_dataset_torch, batch_size=batch_size, shuffle=False) 

    total_test_loss = 0
    target_frames_plotting = []
    prediction_frames_plotting = []

    with torch.no_grad():
        for step, (batch,) in enumerate(flow_loader, start=1):
            batch = batch.to(device)
            B, T, C, H, W = batch.shape
            
            min_clip_len = pred_steps + 1
            if T < min_clip_len:
                print(f"Skipping validation batch: Clip length T={T} < {min_clip_len}")
                continue

            input_len = T - pred_steps
            
            context = batch[:, :input_len]
            target = batch[:, input_len:input_len + pred_steps]

            preds = model(context)

            target_frames_plotting.append(target[:, :, 0])
            prediction_frames_plotting.append(preds[:, :, 0])

            # Loss calculation (using pre-calculated and normalized time_weights)
            per_frame_loss = torch.stack(
                [criterion(preds[:, t], target[:, t]) for t in range(pred_steps)]
            )
            weighted_loss = (per_frame_loss * time_weights_normalized).sum()
            total_test_loss += weighted_loss.item()

        for step, (batch,) in enumerate(test_loader, start=1):
            batch = batch.to(device)
            B, T, C, H, W = batch.shape
            
            min_clip_len = pred_steps + 1
            if T < min_clip_len:
                print(f"Skipping validation batch: Clip length T={T} < {min_clip_len}")
                continue

            input_len = T - pred_steps
            
            context = batch[:, :input_len]
            target = batch[:, input_len:input_len + pred_steps]

            preds = model(context)

            target_frames_plotting.append(target[:, :, 0])
            prediction_frames_plotting.append(preds[:, :, 0])

            # Loss calculation (using pre-calculated and normalized time_weights)
            per_frame_loss = torch.stack(
                [criterion(preds[:, t], target[:, t]) for t in range(pred_steps)]
            )
            weighted_loss = (per_frame_loss * time_weights_normalized).sum()
            total_test_loss += weighted_loss.item()
            
    avg_test_loss = total_test_loss / len(val_loader)

    # --- Reporting and Checkpointing ---

    print(f"Average Test Loss: {total_test_loss:.4f}")
    print("-----------------------------------\n")

    target_frames = torch.cat(target_frames_plotting, dim=0)
    prediction_frames = torch.cat(prediction_frames_plotting, dim=0)

    diffs = target_frames - prediction_frames
    temporal_diffs = prediction_frames[:, 1:] - prediction_frames[:, :-1]

    

    # Create output directory
    output_dir = Path("fig")
    output_dir.mkdir(exist_ok=True)

    n_samples = min(100, target_frames.shape[0])
    n_timesteps = target_frames.shape[1]

    for sample_idx in range(n_samples):
        fig, axes = plt.subplots(4, n_timesteps, figsize=(n_timesteps * 2, 8))
        
        if n_timesteps == 1:
            axes = axes.reshape(-1, 1)
        
        # Row 0: Target frames
        for t in range(n_timesteps):
            img = target_frames[sample_idx, t].cpu().numpy()
            axes[0, t].imshow(img, cmap='gray')
            axes[0, t].set_title(f'Target t={t}')
            axes[0, t].axis('off')
        
        # Row 1: Prediction frames
        for t in range(n_timesteps):
            img = prediction_frames[sample_idx, t].cpu().numpy()
            axes[1, t].imshow(img, cmap='gray')
            axes[1, t].set_title(f'Prediction t={t}')
            axes[1, t].axis('off')
        
        # Row 2: Diffs
        for t in range(n_timesteps):
            img = diffs[sample_idx, t].cpu().numpy()
            axes[2, t].imshow(img, cmap='RdBu_r', vmin=-np.max(np.abs(img)), vmax=np.max(np.abs(img)))
            axes[2, t].set_title(f'Diff t={t}')
            axes[2, t].axis('off')
        
        # Row 3: Temporal diffs (one less timestep)
        for t in range(n_timesteps - 1):
            img = temporal_diffs[sample_idx, t].cpu().numpy()
            axes[3, t].imshow(img, cmap='RdBu_r', vmin=-np.max(np.abs(img)), vmax=np.max(np.abs(img)))
            axes[3, t].set_title(f'Temporal Diff t={t}')
            axes[3, t].axis('off')
        
        # Hide the last column of the temporal diffs row
        axes[3, -1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / f"sample_{sample_idx}_{epoch}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()

    print(f"All figures saved to {output_dir}/")



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

    evaluate(model, epoch+1)

print("Training complete.")

