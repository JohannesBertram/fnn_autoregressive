"""Train FNN-based autoregressive model to predict 5 future frames.

We construct a simplified model (encoder + recurrent core + decoder) that can
take any number of input frames (T_in >= 1) from a sequence and always predict
`pred_steps` future frames autoregressively. The training script randomly
selects an input length per batch to encourage robustness to context size.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from fnn.microns.build import frame_autoregressive_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pred_steps = 6

seq = np.load("input_data/clips_dataset5.npz")["clips"] 
seq = torch.tensor(seq).unsqueeze(2).float() / 255.0  # [N, T, 1, H, W]
seq = seq

dataset = TensorDataset(seq)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = frame_autoregressive_model(pred_steps=pred_steps).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2

for epoch in range(num_epochs):
    for step, (batch,) in enumerate(train_loader, start=1):
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        print(batch.shape)
        assert T >= pred_steps + 1

        # Randomly choose number of input frames (at least 1, at most T - pred_steps)
        max_input = T - pred_steps
        input_len = np.random.randint(1, max_input + 1)
        context = batch[:, :input_len]              # [B, input_len, 1, 64, 64]
        target = batch[:, input_len:input_len + pred_steps]  # [B, pred_steps, 1, 64, 64]
        #print(context)
        #print(context.shape)

        preds = model(context)  # [B, pred_steps, 1, 144, 256]
        # Apply exponential decay to loss for later time steps
        decay = 0.9  # You can adjust this value
        time_weights = torch.tensor([decay ** t for t in range(pred_steps)], device=preds.device)
        # Compute per-frame loss
        per_frame_loss = torch.stack([criterion(preds[:, t], target[:, t]) for t in range(pred_steps)])  # [pred_steps]
        weighted_loss = (per_frame_loss * time_weights).sum() / time_weights.sum()
        loss = weighted_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"Epoch {epoch+1} Step {step}/{len(train_loader)} | input_len={input_len} | Loss {loss.item():.4f}")

    # Optional: save checkpoint at end of epoch
    torch.save(model.state_dict(), f"checkpoints/autoregressive_fnn_epoch_{epoch+1}.pt")

print("Training complete.")

"""# Compare model output with expected output, e.g., by visualizing predicted frames vs. ground truth.
import matplotlib.pyplot as plt

# load model from checkpoint
model.load_state_dict(torch.load(f"checkpoints/autoregressive_fnn_epoch_{2}.pt", map_location=device))
model.eval()
    
random_indices = torch.randperm(seq.size(0))[:10]  # Select 10 random sequences

with torch.no_grad():
    preds = model(seq[random_indices, :pred_steps + 1])  # Use first few sequences for visualization
target = seq[random_indices, -pred_steps:]

def visualize_predictions(preds, target):
    B, T, C, H, W = preds.shape
    for b in range(B):
        for t in range(T):
            plt.subplot(2, T, t + 1)
            plt.imshow(preds[b, t, 0].detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            plt.subplot(2, T, T+ t + 1)
            plt.imshow(target[b, t, 0].detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.show()

visualize_predictions(preds, target)"""