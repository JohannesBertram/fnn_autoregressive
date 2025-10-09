import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
from fnn.microns.build import frame_autoregressive_model

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_steps = 5
batch_size = 16 # Use a slightly larger batch for efficient evaluation
decay = 0.9
checkpoint_dir = "example_checkpoints"

# --- Utility Functions ---

def get_latest_checkpoint(checkpoint_dir):
    """Finds the most recent checkpoint file based on epoch number."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("autoregressive_fnn_epoch_") and f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract epoch number and sort
    epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_epoch = 10
    latest_file = os.path.join(checkpoint_dir, f"autoregressive_fnn_epoch_{latest_epoch}.pt")
    return latest_file, latest_epoch

def calculate_weighted_loss(preds, target, criterion, time_weights_normalized):
    """Calculates the time-weighted MSE loss."""
    # Ensure all tensors are on the same device before stack/mul
    preds = preds.to(time_weights_normalized.device)
    target = target.to(time_weights_normalized.device)

    per_frame_loss = torch.stack(
        [criterion(preds[:, t], target[:, t]) for t in range(pred_steps)]
    )  # [pred_steps]
    weighted_loss = (per_frame_loss * time_weights_normalized).sum()
    return weighted_loss.item()

def load_data_and_split():
    """Loads the data and returns the test_dataset."""
    seq = np.load("input_data/clips_dataset.npz")["clips"] 
    seq = torch.tensor(seq).unsqueeze(2).float() / 255.0  # [N, T, 1, H, W]
    full_dataset = TensorDataset(seq)

    # Use the same split logic as the training script
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    return test_dataset

# --- Main Evaluation Script ---

# 1. Load Data and Checkpoint
print("Loading data and model...")
test_dataset = load_data_and_split()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get the latest checkpoint
try:
    latest_checkpoint_path, latest_epoch = get_latest_checkpoint(checkpoint_dir)
except FileNotFoundError as e:
    print(e)
    exit()

# Initialize model, criterion, and weights
model = frame_autoregressive_model(pred_steps=pred_steps).to(device)
model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))
model.eval()
criterion = torch.nn.MSELoss()

# Pre-calculate time weights for loss calculation
time_weights = torch.tensor([decay ** t for t in range(pred_steps)], device=device)
time_weights_normalized = time_weights / time_weights.sum()

print(f"Loaded checkpoint from Epoch {latest_epoch} at: {latest_checkpoint_path}")
print(f"Test Set Size: {len(test_dataset)}")

# 2. Calculate Test Loss
print("\nCalculating Test Loss...")
total_test_loss = 0.0
num_batches = 0

with torch.no_grad():
    for (batch,) in test_loader:
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        
        min_clip_len = pred_steps + 1
        if T < min_clip_len:
            continue

        # Use the maximum possible input length for consistent evaluation
        input_len = T - pred_steps
        
        context = batch[:, :input_len]
        target = batch[:, input_len:input_len + pred_steps]

        preds = model(context)

        loss = calculate_weighted_loss(preds, target, criterion, time_weights_normalized)
        total_test_loss += loss
        num_batches += 1

avg_test_loss = total_test_loss / num_batches
print(f"Average Weighted Test Loss: {avg_test_loss:.6f}")

# 3. Visualization
print("\nGenerating Visualizations...")

# Grab one batch from the test loader for visualization
vis_batch_tensor = next(iter(test_loader))[0]
vis_batch_tensor = vis_batch_tensor.to(device)
B_vis, T_vis, C_vis, H_vis, W_vis = vis_batch_tensor.shape

# Use the first sequence in the batch for visualization
vis_clip = vis_batch_tensor[3:4] # [1, T, 1, H, W]

# Define input/target split for visualization
vis_input_len = T_vis - pred_steps
vis_context = vis_clip[:, :vis_input_len]
vis_target = vis_clip[:, vis_input_len:vis_input_len + pred_steps]

# Generate prediction
with torch.no_grad():
    vis_preds = model(vis_context)

# Convert tensors to numpy for plotting (and squeeze the batch/channel dims)
context_np = vis_context.squeeze().cpu().numpy() # [input_len, H, W]
target_np = vis_target.squeeze().cpu().numpy()   # [pred_steps, H, W]
preds_np = vis_preds.squeeze().cpu().numpy()     # [pred_steps, H, W]

# Calculate difference in consecutive predictions (Temporal Difference)
# diff[t] = |pred[t+1] - pred[t]| for t in [0, pred_steps-2]
pred_diff_np = np.abs(preds_np[1:] - preds_np[:-1]) # [pred_steps - 1, H, W]

# Determine plot dimensions
num_context_frames = context_np.shape[0]
total_frames = num_context_frames + pred_steps
num_diff_frames = pred_diff_np.shape[0]

# Total columns: Predictions (pred_steps) + Target (pred_steps) + Differences (pred_steps-1)
# We will arrange them into rows:
# Row 1: Predictions (pred_steps)
# Row 2: Target Frames (pred_steps)
# Row 3: Difference Frames (pred_steps-1)
# NOTE: To fit this nicely, let's plot a few key frames.

fig, axes = plt.subplots(
    nrows=3, 
    ncols=pred_steps + 1, # Max columns for better layout
    figsize=(16, 10)
)
plt.suptitle(
    f'Frame Prediction Visualization (Epoch {latest_epoch}) | Test Loss: {avg_test_loss:.4f}', 
    fontsize=14
)

# --- Row 1: Context and Predictions ---
row1_max_cols = axes.shape[1]
# Context Frames
#for i in range(num_context_frames):
#    ax = axes[0, i + 1]
#    ax.imshow(context_np[i], cmap='gray', vmin=0, vmax=1)
#    ax.set_title(f'Input T={i}', color='blue')
#    ax.axis('off')
axes[0, 0].text(0.5, 0.5, 'PREDICTIONS:', ha='center', va='center', fontsize=12, color='green')
axes[0, 0].axis('off')
# Predictions
for i in range(pred_steps):
    ax = axes[0, i + 1]
    ax.imshow(preds_np[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Pred T+{i+1}', color='red')
    ax.axis('off')

# Hide remaining axes in the first row
#for i in range(num_context_frames + pred_steps, row1_max_cols):
#    axes[0, i].axis('off')

# --- Row 2: Ground Truth Target Frames ---
axes[1, 0].text(0.5, 0.5, 'TARGETS:', ha='center', va='center', fontsize=12, color='green')
axes[1, 0].axis('off')

for i in range(pred_steps):
    ax = axes[1, i + 1]
    ax.imshow(target_np[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Target T+{i+1}', color='green')
    ax.axis('off')

# Hide remaining axes in the second row
for i in range(pred_steps + 1, row1_max_cols):
    axes[1, i].axis('off')

# --- Row 3: Difference of Consecutive Predicted Frames ---
axes[2, 0].text(0.5, 0.5, 'DIFF(|P(t+1)-P(t)|):', ha='center', va='center', fontsize=12, color='purple')
axes[2, 0].axis('off')

# Use a colormap for differences (e.g., 'viridis') and a fixed max value
max_diff = np.max(pred_diff_np) * 1.1 if np.max(pred_diff_np) > 0 else 0.1

for i in range(num_diff_frames):
    ax = axes[2, i + 1]
    im = ax.imshow(pred_diff_np[i], cmap='plasma', vmin=0, vmax=max_diff)
    ax.set_title(f'Diff T+{i+1}/{i+2}', color='purple')
    ax.axis('off')

# Add a colorbar for the difference images
if max_diff > 0:
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.25]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

# Hide remaining axes in the third row
for i in range(num_diff_frames + 1, row1_max_cols):
    axes[2, i].axis('off')


plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Adjust layout for suptitle and colorbar
plt.savefig("fig/test_seq.png")

print("Visualization complete.")


def generate_moving_grating(num_frames, H, W, speed=10, direction='horizontal', frequency=0.01, max_val=255.0):
    """
    Generates a sequence of frames with a moving grating pattern.

    Args:
        num_frames (int): Total number of frames in the sequence.
        H (int): Height of each frame.
        W (int): Width of each frame.
        speed (float): How many pixels the grating shifts per frame.
        direction (str): 'horizontal' or 'vertical'.
        frequency (float): Spatial frequency of the grating (controls stripe thickness).
        max_val (float): Maximum pixel value (e.g., 255 for uint8, 1.0 for float).

    Returns:
        np.ndarray: [num_frames, H, W] array of the moving grating sequence.
    """
    grating_sequence = np.zeros((num_frames, H, W), dtype=np.float32)

    if direction == 'horizontal':
        x = np.linspace(0, 2 * np.pi * frequency * W, W)
        for t in range(num_frames):
            shift = t * speed
            # Use modulo to wrap around the pattern
            pattern = np.cos(x + shift * 2 * np.pi * frequency) 
            grating_frame = (pattern + 1) / 2 * max_val # Scale to [0, max_val]
            grating_sequence[t, :, :] = grating_frame
    elif direction == 'vertical':
        y = np.linspace(0, 2 * np.pi * frequency * H, H)
        for t in range(num_frames):
            shift = t * speed
            pattern = np.cos(y + shift * 2 * np.pi * frequency)
            grating_frame = (pattern + 1) / 2 * max_val
            grating_sequence[t, :, :] = grating_frame[:, np.newaxis] # Expand to (H, 1) then broadcast
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    # Normalize to [0, 1] as the model expects
    return grating_sequence / max_val


grating = generate_moving_grating(45, 144, 256)
print(grating.shape)

# 3. Visualization
print("\nGenerating Visualizations...")

vis_clip = torch.Tensor(grating).unsqueeze(0).unsqueeze(2).to(device)

# Define input/target split for visualization
vis_input_len = T_vis - pred_steps
vis_context = vis_clip[:, :vis_input_len]
vis_target = vis_clip[:, vis_input_len:vis_input_len + pred_steps]

# Generate prediction
with torch.no_grad():
    vis_preds = model(vis_context)

# Convert tensors to numpy for plotting (and squeeze the batch/channel dims)
context_np = vis_context.squeeze().cpu().numpy() # [input_len, H, W]
target_np = vis_target.squeeze().cpu().numpy()   # [pred_steps, H, W]
preds_np = vis_preds.squeeze().cpu().numpy()     # [pred_steps, H, W]

# Calculate difference in consecutive predictions (Temporal Difference)
# diff[t] = |pred[t+1] - pred[t]| for t in [0, pred_steps-2]
pred_diff_np = np.abs(preds_np[1:] - preds_np[:-1]) # [pred_steps - 1, H, W]

# Determine plot dimensions
num_context_frames = context_np.shape[0]
total_frames = num_context_frames + pred_steps
num_diff_frames = pred_diff_np.shape[0]

# Total columns: Predictions (pred_steps) + Target (pred_steps) + Differences (pred_steps-1)
# We will arrange them into rows:
# Row 1: Predictions (pred_steps)
# Row 2: Target Frames (pred_steps)
# Row 3: Difference Frames (pred_steps-1)
# NOTE: To fit this nicely, let's plot a few key frames.

fig, axes = plt.subplots(
    nrows=3, 
    ncols=pred_steps + 1, # Max columns for better layout
    figsize=(16, 10)
)
plt.suptitle(
    f'Frame Prediction Visualization (Epoch {latest_epoch}) | Test Loss: {avg_test_loss:.4f}', 
    fontsize=14
)

# --- Row 1: Context and Predictions ---
row1_max_cols = axes.shape[1]
# Context Frames
#for i in range(num_context_frames):
#    ax = axes[0, i + 1]
#    ax.imshow(context_np[i], cmap='gray', vmin=0, vmax=1)
#    ax.set_title(f'Input T={i}', color='blue')
#    ax.axis('off')
axes[0, 0].text(0.5, 0.5, 'PREDICTIONS:', ha='center', va='center', fontsize=12, color='green')
axes[0, 0].axis('off')
# Predictions
for i in range(pred_steps):
    ax = axes[0, i + 1]
    ax.imshow(preds_np[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Pred T+{i+1}', color='red')
    ax.axis('off')

# Hide remaining axes in the first row
#for i in range(num_context_frames + pred_steps, row1_max_cols):
#    axes[0, i].axis('off')

# --- Row 2: Ground Truth Target Frames ---
axes[1, 0].text(0.5, 0.5, 'TARGETS:', ha='center', va='center', fontsize=12, color='green')
axes[1, 0].axis('off')

for i in range(pred_steps):
    ax = axes[1, i + 1]
    ax.imshow(target_np[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Target T+{i+1}', color='green')
    ax.axis('off')

# Hide remaining axes in the second row
for i in range(pred_steps + 1, row1_max_cols):
    axes[1, i].axis('off')

# --- Row 3: Difference of Consecutive Predicted Frames ---
axes[2, 0].text(0.5, 0.5, 'DIFF(|P(t+1)-P(t)|):', ha='center', va='center', fontsize=12, color='purple')
axes[2, 0].axis('off')

# Use a colormap for differences (e.g., 'viridis') and a fixed max value
max_diff = np.max(pred_diff_np) * 1.1 if np.max(pred_diff_np) > 0 else 0.1

for i in range(num_diff_frames):
    ax = axes[2, i + 1]
    im = ax.imshow(pred_diff_np[i], cmap='plasma', vmin=0, vmax=max_diff)
    ax.set_title(f'Diff T+{i+1}/{i+2}', color='purple')
    ax.axis('off')

# Add a colorbar for the difference images
if max_diff > 0:
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.25]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

# Hide remaining axes in the third row
for i in range(num_diff_frames + 1, row1_max_cols):
    axes[2, i].axis('off')


plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Adjust layout for suptitle and colorbar
plt.savefig("fig/grat_seq.png")

print("Visualization complete.")

def generate_split_grating(num_frames, H, W, speed=10, frequency=0.01, max_val=255.0):
    """
    Generates a sequence with split gratings: left half moves up, right half moves down.
    
    Args:
        num_frames (int): Total number of frames in the sequence.
        H (int): Height of each frame.
        W (int): Width of each frame.
        speed (float): How many pixels the grating shifts per frame.
        frequency (float): Spatial frequency of the grating (controls stripe thickness).
        max_val (float): Maximum pixel value (e.g., 255 for uint8, 1.0 for float).
    
    Returns:
        np.ndarray: [num_frames, H, W] array of the split moving grating sequence.
    """
    grating_sequence = np.zeros((num_frames, H, W), dtype=np.float32)
    mid_point = W // 2
    
    # Generate vertical grating pattern
    y = np.linspace(0, 2 * np.pi * frequency * H, H)
    
    for t in range(num_frames):
        frame = np.zeros((H, W), dtype=np.float32)
        
        # Left half: moving upward (negative shift)
        shift_left = -t * speed
        pattern_left = np.cos(y + shift_left * 2 * np.pi * frequency)
        grating_left = (pattern_left + 1) / 2 * max_val
        frame[:, :mid_point] = grating_left[:, np.newaxis]
        
        # Right half: moving downward (positive shift)
        shift_right = t * speed
        pattern_right = np.cos(y + shift_right * 2 * np.pi * frequency)
        grating_right = (pattern_right + 1) / 2 * max_val
        frame[:, mid_point:] = grating_right[:, np.newaxis]
        
        grating_sequence[t, :, :] = frame
    
    # Normalize to [0, 1] as the model expects
    return grating_sequence / max_val


# Generate split grating
split_grating = generate_split_grating(45, 144, 256)
print(f"Split grating shape: {split_grating.shape}")

# Visualization for split grating
print("\nGenerating Split Grating Visualizations...")

vis_clip_split = torch.Tensor(split_grating).unsqueeze(0).unsqueeze(2).to(device)

# Define input/target split for visualization
vis_context_split = vis_clip_split[:, :vis_input_len]
vis_target_split = vis_clip_split[:, vis_input_len:vis_input_len + pred_steps]

# Generate prediction
with torch.no_grad():
    vis_preds_split = model(vis_context_split)

# Convert tensors to numpy for plotting
context_np_split = vis_context_split.squeeze().cpu().numpy()
target_np_split = vis_target_split.squeeze().cpu().numpy()
preds_np_split = vis_preds_split.squeeze().cpu().numpy()

# Calculate difference in consecutive predictions
pred_diff_np_split = np.abs(preds_np_split[1:] - preds_np_split[:-1])

# Create figure
fig, axes = plt.subplots(
    nrows=3, 
    ncols=pred_steps + 1,
    figsize=(16, 10)
)
plt.suptitle(
    f'Split Grating Prediction (Left↑ Right↓) - Epoch {latest_epoch} | Test Loss: {avg_test_loss:.4f}', 
    fontsize=14
)

# Row 1: Predictions
axes[0, 0].text(0.5, 0.5, 'PREDICTIONS:', ha='center', va='center', fontsize=12, color='green')
axes[0, 0].axis('off')
for i in range(pred_steps):
    ax = axes[0, i + 1]
    ax.imshow(preds_np_split[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Pred T+{i+1}', color='red')
    ax.axis('off')

# Row 2: Ground Truth
axes[1, 0].text(0.5, 0.5, 'TARGETS:', ha='center', va='center', fontsize=12, color='green')
axes[1, 0].axis('off')
for i in range(pred_steps):
    ax = axes[1, i + 1]
    ax.imshow(target_np_split[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Target T+{i+1}', color='green')
    ax.axis('off')

for i in range(pred_steps + 1, axes.shape[1]):
    axes[1, i].axis('off')

# Row 3: Differences
axes[2, 0].text(0.5, 0.5, 'DIFF(|P(t+1)-P(t)|):', ha='center', va='center', fontsize=12, color='purple')
axes[2, 0].axis('off')

max_diff_split = np.max(pred_diff_np_split) * 1.1 if np.max(pred_diff_np_split) > 0 else 0.1
num_diff_frames_split = pred_diff_np_split.shape[0]

for i in range(num_diff_frames_split):
    ax = axes[2, i + 1]
    im = ax.imshow(pred_diff_np_split[i], cmap='plasma', vmin=0, vmax=max_diff_split)
    ax.set_title(f'Diff T+{i+1}/{i+2}', color='purple')
    ax.axis('off')

if max_diff_split > 0:
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.25])
    fig.colorbar(im, cax=cbar_ax)

for i in range(num_diff_frames_split + 1, axes.shape[1]):
    axes[2, i].axis('off')

plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
plt.savefig("fig/split_grat_seq.png")

print("Split grating visualization complete.")