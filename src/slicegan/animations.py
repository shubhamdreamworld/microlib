import asyncio
import logging
import os
import shutil
import time
from scipy import ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
import moviepy.editor as mp
from moviepy.editor import ImageClip, concatenate_videoclips
from src.slicegan import networks, util

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using NVIDIA CUDA GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Metal
        logging.info("Using Apple Metal GPU.")
    elif torch.has_mps:  # Alternative check for Metal
        device = torch.device("mps")
        logging.info("Using Apple Metal GPU.")
    else:
        device = torch.device("cpu")
        logging.warning("No compatible GPU found. Using CPU instead.")
    return device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = check_gpu()

class Animator():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.img_display = None

    def new_animation(self, micro):
        self.Project_name = micro
        self.Project_path = f'data/slicegan_runs/{micro}'
        os.makedirs(f'{self.Project_path}/frames', exist_ok=True)
        self.frame_path = f'{self.Project_path}/frames'
        self.Project_path = f'data/slicegan_runs/{micro}/{micro}'
        
        img_size, img_channels, scale_factor = 64, 1, 1
        z_channels = 16
        lays = 6
        dk, gk = [4] * lays, [4] * lays
        ds, gs = [2] * lays, [2] * lays
        df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 64, img_channels]
        dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]
        
        netD, netG = networks.slicegan_nets(self.Project_path, False, 'grayscale', dk, ds, df, dp, gk, gs, gf, gp)
        netG = netG().to(device).eval()
        noise = torch.randn(1, z_channels, 12, 12, 12, device=device)
        netG.load_state_dict(torch.load(self.Project_path + '_Gen.pt', map_location=device))
        
        img = netG(noise[0].unsqueeze(0)).cpu().detach().numpy()
        img = util.post_proc(img, 'grayscale')
        self.img = img
        self.n = img.shape[0]
        self.f = 0
        
        self.update_frame()

    def update_frame(self):
        self.f += 1
        if self.f >= self.n:
            self.f = 0
        img = self.img[self.f]
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        plt.axis('off')
        frame_path = os.path.join(self.frame_path, f'frame_{self.f:05d}.png')
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved frame {frame_path}")

    def save_animation(self):
        frames = sorted(os.listdir(self.frame_path))
        clips = [ImageClip(os.path.join(self.frame_path, m)).set_duration(1/45) for m in frames]
        clip = concatenate_videoclips(clips, method="compose")
        clip.write_videofile(f'{self.Project_path}_animation.mp4', fps=45, logger=None)


def animate_dataset():
    dir = 'data/slicegan_runs'
    micros = sorted(os.listdir(dir))
    logger.info(f"Found {len(micros)} projects.")
    a = Animator()
    for micro in micros:
        a.new_animation(micro)
        for _ in range(a.n):
            a.update_frame()
        a.save_animation()
        logger.info(f"Animation completed for {micro}")
