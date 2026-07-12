import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision.io import decode_image
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
        image = decode_image(img_path)

        mat_name = 'GT_' + os.path.splitext(fname)[0] + '.mat'
        mat = sio.loadmat(os.path.join(self.gt_dir, mat_name))

        points = mat['image_info'][0, 0]['location'][0, 0]
        H, W = image.shape[1], image.shape[2]
        label = self.adaptive_density_map(points, (H,W), k=3, beta=0.3, min_sigma=1, max_sigma=15)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def adaptive_density_map(points, img_shape, k=3, beta=0.3, min_sigma=1, max_sigma=15):
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

            #placeholder map
            pt_map = np.zeros((H, W), dtype=np.float32)
            pt_map[yi, xi] = 1

            if n_points > 1:
                #average distance to nearest neighbors (excluding self at index 0)
                avg_dist = distances[i, 1:].mean() if k_query > 1 else distances[i]
                sigma = np.clip(beta * avg_dist, min_sigma, max_sigma)
            else:
                sigma = max_sigma #for isolated points

            density += gaussian_filter(pt_map, sigma=sigma)

        return density
