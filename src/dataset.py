import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
import torch
from scipy.spatial import KDTree

class CustomImageDataset(Dataset):
    def __init__(self, gt_dir, img_dir, sigma, k, beta, transform=None, target_transform=None,
                 adaptive=True):
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.img_filenames = sorted(                            
            f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        )
        self.transform = transform
        self.target_transform = target_transform
        
        #for converting points to density map 
        self.adaptive = adaptive
        self.sigma = sigma
        self.k = k
        self.beta = beta 

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        fname = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = decode_image(img_path, mode=ImageReadMode.RGB).float() / 255.0

        mat_name = 'GT_' + os.path.splitext(fname)[0] + '.mat'
        mat = sio.loadmat(os.path.join(self.gt_dir, mat_name))

        points = mat['image_info'][0, 0]['location'][0, 0]
        H, W = image.shape[1], image.shape[2]
        label = self.adaptive_density_map(points, (H,W), k=3, beta=0.3, min_sigma=1, max_sigma=15)

        #CSRNet's frontend downsamples by 8x (3 maxpool layers), so the label
        #must be downsampled to match the model's output resolution
        label = self.downsample_density(label, factor=8)
        label = torch.from_numpy(label).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def adaptive_density_map(self, points, img_shape, k=3, beta=0.3, min_sigma=1, max_sigma=15):
        """
        points: (N, 2) array of (x, y) coordinates
        img_shape: (H, W)
        k: number of nearest neighbors to consider
        beta: scaling factor for neighbor distance -> sigma
        """
        H, W = img_shape
        density = np.zeros((H, W), dtype=np.float32)
        n_points = len(points)

        if n_points == 0:
            return density

        #KD-tree for nearest-neighbor lookup
        tree = KDTree(points)
        k_query = min(k + 1, n_points)
        distances, _ = tree.query(points, k=k_query)

        for i, (x, y) in enumerate(points):
            xi, yi = int(x), int(y)
            if not (0 <= yi < H and 0 <= xi < W):
                continue

            if n_points > 1:
                #average distance to nearest neighbors (excluding self at index 0)
                avg_dist = distances[i, 1:].mean() if k_query > 1 else distances[i]
                sigma = np.clip(beta * avg_dist, min_sigma, max_sigma)
            else:
                sigma = max_sigma #for isolated points

            #only compute the Gaussian over a small patch around the point instead
            #of blurring the full image (which is O(H*W) per point and was the
            #main training bottleneck for images with many annotations)
            radius = max(1, int(np.ceil(3 * sigma)))
            kernel = self._gaussian_kernel(sigma, radius)

            y_min, y_max = yi - radius, yi + radius + 1
            x_min, x_max = xi - radius, xi + radius + 1

            ky_min, ky_max = max(0, -y_min), kernel.shape[0] - max(0, y_max - H)
            kx_min, kx_max = max(0, -x_min), kernel.shape[1] - max(0, x_max - W)

            density[max(0, y_min):min(H, y_max), max(0, x_min):min(W, x_max)] += \
                kernel[ky_min:ky_max, kx_min:kx_max]

        return density

    def _gaussian_kernel(self, sigma, radius):
        ax = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
        kernel /= kernel.sum()
        return kernel

    def downsample_density(self, density, factor=8):
        """
        Sum-pools the density map by `factor` so total count is preserved,
        matching CSRNet's downsampled output resolution.
        """
        H, W = density.shape
        H_ds, W_ds = H // factor, W // factor
        density = density[:H_ds * factor, :W_ds * factor]
        density = density.reshape(H_ds, factor, W_ds, factor).sum(axis=(1, 3))
        return density
