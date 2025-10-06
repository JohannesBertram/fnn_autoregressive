import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_and_crop_mp4(
    mp4_path,
    resize=(256, 144),
    diff_threshold=0.15,
    min_clip_length=10,
    crop_length=10,
    save_path="clips_dataset.npz"
):
    """
    Reads an MP4 file (60 fps assumed), downsamples it to ~30 fps using frame
    averaging, segments it into clips by detecting abrupt transitions,
    generates all crops of length `crop_length` from each clip, and saves the result.

    Parameters
    ----------
    mp4_path : str
        Path to the MP4 file.
    resize : tuple
        (width, height) to resize frames.
    diff_threshold : float
        Threshold for normalized mean absolute frame difference to detect cuts.
    min_clip_length : int
        Minimum number of *30 fps* frames for a valid clip.
    crop_length : int
        Length of crops (in *30 fps* frames) to extract from each clip.
    save_path : str
        Path to save the resulting dataset (.npz).
    """
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    diff_frames = []
    counter = 0

    while counter < 1000000:
        # Read the first frame of the pair
        ret1, frame1 = cap.read()
        
        # Read the second frame of the pair
        ret2, frame2 = cap.read()

        # Break if we can't read a full pair (or if we hit the counter limit)
        if not ret1 or not ret2:
            break

        # Process the pair
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute the average frame (now 30 fps)
        # We use .astype(float) to allow for averaging, then round and cast back to uint8
        #averaged_frame = np.round((gray1.astype(float) + gray2.astype(float)) / 2.0).astype(np.uint8)
        averaged_frame = gray1.astype(np.uint8)

        resized = cv2.resize(averaged_frame, resize)
        
        # Use variance check on the new averaged frame
        if np.var(resized>0.1): # remove black/gray screen
            frames.append(resized)
            
            # Also resize for difference calculation (maintaining ~30 fps)
            resized_diff = cv2.resize(averaged_frame, (resize[0]//4, resize[1]//4))
            diff_frames.append(resized_diff)

        # Increment counter by 2 frames (since we read a pair)
        counter += 2
        
    cap.release()
    frames = np.array(frames, dtype="uint8")  # [N/2, H, W] (approx 30 fps)
    diff_frames = np.array(diff_frames, dtype="uint8")

    # The rest of the logic remains the same, operating on the now-30-fps streams

    # Compute normalized frame differences
    diffs = np.mean(np.abs(np.diff(diff_frames, axis=0)), axis=(1,2)) / 255.0  # [N/2 - 1]

    plt.hist(diffs, bins=30)
    plt.savefig("hist.png")
    plt.close()

    # Find cut points (where diff > threshold)
    # The cut indices now refer to the downsampled (30 fps) frames
    cut_indices = np.where(diffs > diff_threshold)[0] + 1  # +1: cut after this frame
    # Add start and end
    boundaries = np.concatenate(([0], cut_indices, [len(frames)]))

    # Extract clips and take all crops (if len(clip) > crop_length)
    clips = []
    for i in range(len(boundaries)-1):
        start, end = boundaries[i], boundaries[i+1]
        clip = frames[start:end]  # [clip_length_30fps, H, W]
        
        # Original logic: only take clips longer than min_clip_length
        # and then only take the very first crop_length frames
        # if len(clip) < min_clip_length:
        #     continue
        # clips.append(clip[:crop_length])
        
        # Revised logic for generating *all* crops of length `crop_length`
        # as described in the docstring's implied intent:
        
        if len(clip) >= min_clip_length:
            # Generate all possible crops of length `crop_length`
            for j in range(0, len(clip) - crop_length + 1, crop_length):
                crop = clip[j:j + crop_length]
                clips.append(crop)


    if not clips:
        # Handle the case where no clips are found
        print(f"Warning: No valid clips of length {min_clip_length} (30 fps) were found.")
        return np.array([], dtype="uint8") # Return an empty array

    clips = np.stack(clips, axis=0)  # [n_crops, crop_length, H, W]
    # Save as compressed npz
    #np.savez_compressed(save_path, clips=clips)
    print(f"Loaded {clips.shape[0]} crops of shape {clips.shape[1:]} to {save_path}")
    return clips

#segment_and_crop_mp4(
#    "input_data/stimulus_17797_4_7_v4_compressed.mp4",
#    resize=(256, 144),
#    diff_threshold=0.55,
#    min_clip_length=45,
#    crop_length=45,
#    save_path="input_data/clips_dataset_47.npz"
#)