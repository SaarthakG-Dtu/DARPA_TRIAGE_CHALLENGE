import torch
import cv2
import argparse
import utils.saveload
import utils.improc
import utils.basic
import PIL.Image
import numpy as np
import os
import time
from prettytable import PrettyTable
import matplotlib.pyplot as plt



def plot_all_trajectories_on_frame(trajs, base_frame, save_path="all_trajectories_overlay.png"):
    if torch.is_tensor(trajs): trajs = trajs.detach().cpu().numpy()
    T, N, _ = trajs.shape
    plt.figure(figsize=(10, 8)); plt.imshow(base_frame)
    for i in range(N):
        plt.plot(trajs[:, i, 0], trajs[:, i, 1], marker='o', markersize=2, linewidth=1.2)
    plt.title("All Trajectories on Frame"); plt.axis('off')
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Overlay of all trajectories saved to {save_path}")

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

def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
    device = rgbs.device; T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2); visibs = visibs.permute(1,0) # N,T,2 and N,T
    N = trajs.shape[0]; colors = torch.tensor(colormap, dtype=torch.float32, device=device)
    rgbs = rgbs * bkg_opacity; radius = 1 if rate <= 2 else (2 if rate <= 4 else (4 if rate <= 8 else 6))
    sharpness = 0.15 + 0.05 * np.log2(rate); D = radius * 2 + 1
    y, x = torch.meshgrid(torch.arange(D, device=device), torch.arange(D, device=device), indexing="ij")
    y, x = y.float() - radius, x.float() - radius; dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1).view(1, D, D)
    dx, dy = torch.arange(-radius, radius + 1, device=device), torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")
    for t in range(T):
        mask = visibs[:, t];
        if mask.sum() == 0: continue
        xy = trajs[mask, t] + 0.5; xy[:, 0].clamp_(0, W - 1); xy[:, 1].clamp_(0, H - 1)
        colors_now = colors[mask]; N_now = xy.shape[0]; cx, cy = xy[:, 0].long(), xy[:, 1].long()
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

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"]); total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        if param > 100000: table.add_row([name, param])
        total_params += param
    print(table); print('total params: %.2f M' % (total_params / 1e6)); return total_params

def forward_video(rgbs, framerate, model, args):
    B, T, C, H, W = rgbs.shape
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float().permute(0, 2, 1).reshape(1, 1, 2, H, W)
    torch.cuda.empty_cache()
    print('Starting forward pass...')
    
    with torch.no_grad():
        flows_e, visconf_maps_e, _, _ = model(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None)
        traj_maps_e = flows_e + grid_xy
        if args.query_frame > 0:
            backward_flows_e, backward_visconf_maps_e, _, _ = model(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None)
            backward_traj_maps_e = (backward_flows_e + grid_xy).flip([1])[:, :-1]
            backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1]
            traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
            visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1)

    rate = args.rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B, T, 2, -1).permute(0, 1, 3, 2)
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B, T, 2, -1).permute(0, 1, 3, 2)
    
    # Save trajectories
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "trajectories.npy")
    np.save(output_path, trajs_e[0].cpu().numpy())
    print(f"Saved raw trajectories to {output_path}")

    # Visualization
    xy0 = trajs_e[0, 0].cpu().numpy(); colors = utils.improc.get_2d_colors(xy0, H, W)
    fn = args.mp4_path.split('/')[-1].split('.')[0]
    rgb_out_f = os.path.join(args.output_dir, f'pt_vis_{fn}_rate{rate}_q{args.query_frame}.mp4')
    frames = draw_pts_gpu(rgbs[0].cuda(), trajs_e[0], visconfs_e[0,:,:,1] > args.conf_thr, colors, rate=rate)
    
    temp_dir = os.path.join(args.output_dir, "temp_frames")
    utils.basic.mkdir(temp_dir)
    for ti, frame in enumerate(frames):
        PIL.Image.fromarray(frame).save(os.path.join(temp_dir, f'{ti:04d}.jpg'))
    
    os.system(f'/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {framerate} -i "{temp_dir}/%04d.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p {rgb_out_f}')
    print(f"Visualization video saved to {rgb_out_f}")
    shutil.rmtree(temp_dir)

def run(model, args):
    if args.ckpt_init:
        utils.saveload.load(None, args.ckpt_init, model)
        print('Loaded weights from', args.ckpt_init)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('Loaded weights from HuggingFace.')

    model.cuda().eval()
    rgbs, framerate = read_mp4(args.mp4_path, target_fps=10)
    
    if args.max_frames: rgbs = rgbs[:args.max_frames]

    H_orig, W_orig = rgbs[0].shape[:2]
    HH = 384; scale = min(HH / H_orig, HH / W_orig)
    H, W = int(H_orig * scale) // 8 * 8, int(W_orig * scale) // 8 * 8
    rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    
    rgbs = torch.stack([torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgbs]).unsqueeze(0).float().cuda()
    
    forward_video(rgbs, framerate, model, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4_path", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="alltracker_output", help="Directory to save trajectories and visualizations")
    parser.add_argument("--query_frame", type=int, default=16, help="Which frame to track from")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--rate", type=int, default=2, help="Subsampling rate for visualization")
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--window_len", type=int, default=16)
    parser.add_argument("--conf_thr", type=float, default=0.1)
    parser.add_argument("--ckpt_init", type=str, default='')
    
    args = parser.parse_args()
    from nets.alltracker import Net
    import shutil
    model = Net(args.window_len)
    count_parameters(model)
    run(model, args)