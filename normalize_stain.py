import os
import numpy as np
from skimage import io
from PIL import Image
from histomicstk.saliency.tissue_detection import get_tissue_mask
from histomicstk.preprocessing.color_normalization import reinhard
from skimage.transform import resize


class StainNormalizer:
    def __init__(self):
        self.cnorm = {
            'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
            'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
        }

    def normalize_stain(self, slide_path, input_subdir):
        tissue_rgb = io.imread(slide_path)
        if tissue_rgb.shape[2] == 4:
            tissue_rgb = tissue_rgb[:, :, :3]
        tissue_from_array = Image.fromarray(tissue_rgb)
        tissue_copy = tissue_from_array.copy()
        tissue_copy.thumbnail((tissue_rgb.shape[0] / 3, tissue_rgb.shape[1] / 3))
        mask_out, _ = get_tissue_mask(
            np.array(tissue_copy), deconvolve_first=True,
            n_thresholding_steps=1, sigma=1.5, min_size=30)
        mask_out = resize(
            mask_out == 0, output_shape=tissue_rgb.shape[:2],
            order=0, preserve_range=True) == 1
        tissue_rgb_normalized = reinhard(
            tissue_rgb, target_mu=self.cnorm['mu'], target_sigma=self.cnorm['sigma'], mask_out=mask_out)
        tissue_from_array = Image.fromarray(tissue_rgb_normalized)
        save_dir = os.path.join(input_subdir, os.path.splitext(os.path.basename(slide_path))[0] + '_normalized.tiff')
        tissue_from_array.save(save_dir, compression="tiff_adobe_deflate")
        print("{} Staining normalisation completed.".format(slide_path))


def process_directory(data_root, output_root):
    normalizer = StainNormalizer()
    
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_root, folder_name)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.tiff'):
                    slide_path = os.path.join(folder_path, file_name)
                    normalizer.normalize_stain(slide_path, output_folder)


if __name__ == '__main__':
    data_root = 'DATA_RAW_DIRECTORY'
    output_root = 'DATA_DIRECTORY'

    process_directory(data_root, output_root)
