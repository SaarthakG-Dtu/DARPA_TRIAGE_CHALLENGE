import torch
import cv2
import argparse
import utils.improc
import utils.basic
import PIL.Image
import numpy as np
import os
import shutil

def read_mp4(name_path, target_fps=10):
    vidcap = cv2.VideoCapture(name_path)
    original_fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Original framerate:', original_fps, 'Total frames:', total_frames)
    duration_secs = total_frames / original_fps
    target_frame_count = int(duration_secs * target_fps)
    print(f"Sampling {target_frame_count} frames to match {target_fps} FPS over {duration_secs:.2f}s")
    all_frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret: break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vidcap.release()
    indices = np.linspace(0, len(all_frames) - 1, target_frame_count).astype(int)
    frames = [all_frames[i] for i in indices]
    return frames, target_fps

def draw_pts_gpu(rgbs, trajs, twitch_predictions, rate=1, bkg_opacity=0.5):
    device = rgbs.device; T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2); # N,T,2
    N = trajs.shape[0]

    # --- MINIMAL CHANGE IS HERE ---
    # Create a base colormap
    xy0 = trajs[:, 0, :].cpu().numpy()
    colors_base = utils.improc.get_2d_colors(xy0, H, W)
    colors = torch.tensor(colors_base, dtype=torch.float32, device=device)
    
    # If we have predictions, set twitch trajectories to white
    if twitch_predictions is not None and len(twitch_predictions) == N:
        twitch_indices = torch.from_numpy(twitch_predictions).bool().to(device)
        colors[twitch_indices] = torch.tensor([255.0, 255.0, 255.0], dtype=torch.float32, device=device)
    # ----------------------------

    rgbs = rgbs * bkg_opacity; radius = 1 if rate <= 2 else (2 if rate <= 4 else (4 if rate <= 8 else 6))
    sharpness = 0.15 + 0.05 * np.log2(rate); D = radius * 2 + 1
    y, x = torch.meshgrid(torch.arange(D, device=device), torch.arange(D, device=device), indexing="ij")
    y, x = y.float() - radius, x.float() - radius; dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1).view(1, D, D)
    dx, dy = torch.arange(-radius, radius + 1, device=device), torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    for t in range(T):
        if t >= trajs.shape[1]: continue
        
        xy = trajs[:, t] + 0.5; xy[:, 0].clamp_(0, W - 1); xy[:, 1].clamp_(0, H - 1)
        colors_now = colors; N_now = xy.shape[0]; cx, cy = xy[:, 0].long(), xy[:, 1].long()
        x_grid, y_grid = cx[:, None, None] + disp_x, cy[:, None, None] + disp_y
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid, y_valid, icon_weights = x_grid[valid], y_grid[valid], icon.expand(N_now, D, D)[valid]
        colors_valid = colors_now[:, :, None, None].expand(N_now, 3, D, D).permute(1, 0, 2, 3)[:, valid]
        idx_flat = (y_valid * W + x_valid).long()
        accum, weight = torch.zeros_like(rgbs[t]), torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), colors_valid * icon_weights)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W); alpha = weight.clamp(0, 1) * 1.0
        accum = accum / (weight + 1e-6); rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
        
    return rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"[INFO] Loading video from: {args.mp4_path}")
    rgbs, framerate = read_mp4(args.mp4_path, target_fps=10)
    
    print(f"[INFO] Loading filtered trajectories from: {args.trajs_file}")
    filtered_trajs = np.load(args.trajs_file)
    
    print(f"[INFO] Loading predictions from: {args.predictions_file}")
    predictions = np.load(args.predictions_file)

    if filtered_trajs.shape[1] != len(predictions):
        print(f"[ERROR] Mismatch between number of trajectories ({filtered_trajs.shape[1]}) and predictions ({len(predictions)}).")
        return
        
    H_orig, W_orig = rgbs[0].shape[:2]
    HH = 384; scale = min(HH / H_orig, HH / W_orig)
    H, W = int(H_orig * scale) // 8 * 8, int(W_orig * scale) // 8 * 8
    rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    
    rgbs_torch = torch.stack([torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgbs]).float().to(device)
    trajs_torch = torch.from_numpy(filtered_trajs).float().to(device)

    frames = draw_pts_gpu(rgbs_torch, trajs_torch, predictions, rate=args.rate)
    
    temp_dir = os.path.join(os.path.dirname(args.output_video_path), "temp_frames_highlighted")
    utils.basic.mkdir(temp_dir)
    for ti, frame in enumerate(frames):
        PIL.Image.fromarray(frame).save(os.path.join(temp_dir, f'{ti:04d}.jpg'))
    
    os.system(f'/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {framerate} -i "{temp_dir}/%04d.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p "{args.output_video_path}"')
    print(f"\n[COMPLETE] Highlighted video saved to {args.output_video_path}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a highlighted video of twitch predictions.")
    parser.add_argument("--mp4_path", type=str, required=True, help="Input video path (the one fed to All-Tracker).")
    parser.add_argument("--trajs_file", type=str, required=True, help="Path to the FILTERED trajectories .npy file.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to the saved predictions .npy file.")
    parser.add_argument("--output_video_path", type=str, required=True, help="Path to save the final highlighted .mp4 video.")
    parser.add_argument("--rate", type=int, default=2, help="The subsampling rate used in All-Tracker.")
    args = parser.parse_args()
    main(args)