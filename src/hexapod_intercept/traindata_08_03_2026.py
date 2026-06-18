import sys
sys.path.append(r"C:\Users\annam\OneDrive - South East Technological University\!Year 4\!Project\Libraries")
sys.path.append(r"C:\Users\annam\OneDrive - South East Technological University\!Year 4\!Project\Object Intercept")

import cv2 as cv
import numpy as np
import torch
import random
import os
import Tracking as T

# --- CONFIGURATION ---
video_path    = "Datasets\\Bug_walk_8_03_2026\\"
cropped_path  = "Datasets\\Bug_walk_8_03_2026\\Cropped\\"
TARGET_W      = 1280 
TARGET_H      = 720
TARGET_FPS    = 30
History_length    = 30
Prediction_length = 30
TRAIN_SPLIT   = 0.8
DEBUG = 0
FILTER = "grey"
# ----------------------

os.makedirs(cropped_path, exist_ok=True)
tracker = T.Tracking()

# Build video list
videos = [
    "bug (1).mp4",
    "bug (2).mp4",
    "bug (3).mp4",
    "bug (4).mp4",
    "bug (5).mp4",
    "bug (6).mp4",
    "bug (7).mp4",
    "bug (8).mp4",
    "bug (9).mp4",
    "bug (10).mp4",
    "bug (11).mp4",
    "bug (12).mp4",
    "bug (13).mp4",
    "bug (14).mp4",
    "bug (15).mp4",
]

# --- Train / Test split ---
random.shuffle(videos)
split        = int(round(len(videos) * TRAIN_SPLIT,0))
train_videos = videos[:split]
test_videos  = videos[split:]
print(f"Train: {len(train_videos)} videos | Test: {len(test_videos)} videos")
print("Train videos",train_videos)
print("Test videos",test_videos)

def augment_samples(samples, noise_copies=2, noise_std=0.01):
    """
    For every sample produces 12 total versions:
      original, x-mirror, y-mirror, xy-mirror  × (1 clean + 2 noisy)
 
    Features layout: [vx, vy, ax, ay, speed, angle, v_angle]
    x-mirror : flip vx, ax, angle, v_angle        — mirror left/right
    y-mirror : flip vy, ay, angle, v_angle        — mirror up/down
    xy-mirror: flip vx, vy, ax, ay                — rotate 180 degrees
    Noise is added to inputs only — outputs and positions stay clean.
    """
    augmented = []
 
    def emit(pos, out, inp):
        augmented.append((pos, out.tolist(), inp.tolist()))
        for _ in range(noise_copies):
            noisy = inp + np.random.randn(*inp.shape).astype(np.float32) * noise_std
            augmented.append((pos, out.tolist(), noisy.tolist()))
 
    for positions, output, input_ in samples:
        inp = np.array(input_, dtype=np.float32)
        out = np.array(output, dtype=np.float32)
 
        pf, pc = positions  # pos_future, pos_cur
 
        # --- x-mirror (flip left/right) ---
        inp_x = inp.copy()
        inp_x[:, 0] *= -1   # vx
        inp_x[:, 2] *= -1   # ax
        inp_x[:, 5] *= -1   # angle
        inp_x[:, 6] *= -1   # v_angle
        out_x = out.copy()
        out_x[:, 0] *= -1   # vx

        pf, pc = positions

        pos_x  = ([[TARGET_W - p[0], p[1]]  for p in pf], [TARGET_W - pc[0], pc[1]])
        pos_y  = ([[p[0], TARGET_H - p[1]]  for p in pf], [pc[0], TARGET_H - pc[1]])
        pos_xy = ([[TARGET_W - p[0], TARGET_H - p[1]] for p in pf], [TARGET_W - pc[0], TARGET_H - pc[1]])
 
        # --- y-mirror (flip up/down) ---
        inp_y = inp.copy()
        inp_y[:, 1] *= -1   # vy
        inp_y[:, 3] *= -1   # ay
        inp_y[:, 5] *= -1   # angle
        inp_y[:, 6] *= -1   # v_angle
        out_y = out.copy()
        out_y[:, 1] *= -1   # vy
 
        # --- xy-mirror (rotate 180) ---
        inp_xy = inp.copy()
        inp_xy[:, 0] *= -1  # vx
        inp_xy[:, 1] *= -1  # vy
        inp_xy[:, 2] *= -1  # ax
        inp_xy[:, 3] *= -1  # ay
        # speed unchanged, angle shifts by pi but wraps — leave as-is
        out_xy = out.copy()
        out_xy[:, 0] *= -1  # vx
        out_xy[:, 1] *= -1  # vy
 
        emit(positions, out,    inp)
        emit(pos_x,     out_x,  inp_x)
        emit(pos_y,     out_y,  inp_y)
        emit(pos_xy,    out_xy, inp_xy)
 
    return augmented


def process_video(video_full_path, video_name):
    Vid = cv.VideoCapture(video_full_path)

    if not Vid.isOpened():
        print(f"Error: Could not open {video_full_path}")
        return []

    source_fps = Vid.get(cv.CAP_PROP_FPS)

    # --- Find arena matrix from first valid frame ---
    matrix = None
    while True:
        ret, frame = Vid.read()
        if not ret or frame is None:
            print(f"Could not find arena in {video_full_path}")
            Vid.release()
            return []

        matrix = tracker.Crop_Arena(frame, TARGET_W, TARGET_H)
        if matrix is not None:
            break

    # --- Read and crop remaining frames ---
    raw_frames = []
    while True:
        ret, frame = Vid.read()
        if not ret or frame is None:
            break

        frame = cv.warpPerspective(frame, matrix, (TARGET_W, TARGET_H), flags=cv.INTER_CUBIC)
        frame = cv.resize(frame, (TARGET_W, TARGET_H), interpolation=cv.INTER_AREA)
        raw_frames.append(frame)

    Vid.release()

    if len(raw_frames) == 0:
        return []

    # # --- Normalise to TARGET_FPS ---
    # total    = len(raw_frames)
    # duration = total / source_fps
    # target_n = int(duration * TARGET_FPS)

    # frames = []
    # for i in range(target_n):
    #     src_idx = min(int(i * source_fps / TARGET_FPS), total - 1)
    #     frames.append(raw_frames[src_idx])

    # --- Save cropped video ---
    save_name = video_name.replace(".mp4", "_cropped.mp4")
    save_path = cropped_path + save_name
    fourcc    = cv.VideoWriter_fourcc(*'mp4v')
    writer    = cv.VideoWriter(save_path, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))

    frames = raw_frames
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Saved cropped video: {save_path}")

    # --- Extract centres ---
    centers = []
    for frame in frames:
        
        found,mask = tracker.Contours(frame,FILTER,DEBUG)

        if len(found) == 0:
            continue

        cx, cy = found[0]
        centers.append((cx, cy))

        # Debug
        if DEBUG:
            cv.circle(frame, (int(cx), int(cy)), 3,  (255, 255, 0), -1)  
            cv.imshow("frame",frame)
            cv.imshow("mask",mask)
            if cv.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    return centers


def build_samples(centers_list):
    samples = []

    for centers in centers_list:
        if len(centers) < 2:  # Need at least 2 points for one velocity
            continue

        centr = np.array(centers, dtype=np.float32)

        Velocities    = np.diff(centr, axis=0)                          # (N-1, 2)
        Accelerations = np.diff(Velocities, axis=0)
        Accelerations = np.vstack([np.zeros((1, 2)), Accelerations]) # (N-1, 2)

        speed   = np.sqrt(Velocities[:,0]**2 + Velocities[:,1]**2).reshape(-1,1)  # (N-1, 1)
        angle   = np.arctan2(Velocities[:,1], Velocities[:,0]).reshape(-1,1)       # (N-1, 1)
        v_angle = np.diff(angle, axis=0)                                            # (N-2, 1)
        v_angle = np.vstack([np.zeros((1, 1)), v_angle])                           # (N-1, 1)  pad front

        features = np.hstack([Velocities, Accelerations, speed, angle, v_angle])   # (N-1, 7)

        # Start from index 1 so we have at least one velocity/accel
        for i in range(1, len(centr) - Prediction_length):
            # Grab however many features exist up to this point
            available = features[:i]  # shape: (i, 4)

            # Pad the front with zeros to reach History_length
            if len(available) >= History_length:
                Input = available[-History_length:]          # full window, no padding
            else:
                pad   = np.zeros((History_length - len(available), features.shape[1]))
                Input = np.vstack([pad, available])          # zeros first, then real data

            Output    = Velocities[i : i + Prediction_length]
            if len(Output) < Prediction_length:
                continue  # Not enough future frames
            
            posCur    = centr[i].tolist()
            posFuture = centr[i + 1 : i + 1 + Prediction_length].tolist()
            positions = (posFuture, posCur)
            samples.append((positions, Output.tolist(), Input.tolist()))

    return samples


def save_split(split_name, video_list):
    print(f"\nProcessing {split_name} split...")

    centers_list = []
    for video in video_list:
        print(f"  {video}")
        centers = process_video(video_path + video, video)
        
        if len(centers) > 0:
            centers_list.append(centers)
        else:
            print(f"  WARNING: no centres found for {video}")

    samples = build_samples(centers_list)

    if split_name == "train":
        samples = augment_samples(samples)
        print(f"  After augmentation: {len(samples)}")

    if len(samples) == 0:
        print(f"  No samples generated for {split_name}")
        return

    last_input  = np.array(samples[-1][2])
    last_output = np.array(samples[-1][1])

    os.makedirs("models", exist_ok=True)
    save_path = f"models/TrainData_{split_name}.pt"

    torch.save({
        "training_pairs":    samples,
        "History_length":    History_length,
        "Prediction_length": Prediction_length,
        "Features":          last_input.shape[1],
        "Output":            last_output.shape[1],
    }, save_path)

    print(f"  Saved {save_path} — {len(samples)} samples")


save_split("train", train_videos)
save_split("test",  test_videos)

cv.destroyAllWindows()
print("\nDone.")