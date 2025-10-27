import os
import argparse
import pandas as pd
import numpy as np
import torch
import openslide

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from models import get_encoder


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_h5_path, output_pt_path, loader, model, verbose = 0):

	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			torch.save(torch.from_numpy(features), output_pt_path)
                  
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_h5_path, asset_dict, attr_dict= None, mode=mode)
			print(f"output_path: {output_h5_path}, shape: {features.shape}")                  
			mode = 'a'


def extract_patient_features(patient_id, wsi_types, data_slide_dir, coords_dir, feat_dir, batch_size, model_name, target_patch_size):

    for wsi_type in wsi_types:
        masker_feat_dir = os.path.join(feat_dir, wsi_type)
        os.makedirs(os.path.join(masker_feat_dir, 'h5_files'), exist_ok=True)
        os.makedirs(os.path.join(masker_feat_dir, 'pt_files'), exist_ok=True)
        output_h5_path = os.path.join(masker_feat_dir, 'h5_files', f"{patient_id}.h5")
        output_pt_path = os.path.join(masker_feat_dir, 'pt_files', f"{patient_id}.pt")

        if f"{patient_id}.pt" in os.listdir(os.path.join(masker_feat_dir, 'pt_files')):
            print(f"Skipping patient {patient_id}_{wsi_type} as features already extracted.")
            continue
        print(f"Processing {wsi_type} for patient {patient_id}")

        model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
        model = model.to(device)
        model.eval()

        slide_id = f"{patient_id}"
        h5_file_path = os.path.join(coords_dir, f"{patient_id}.h5")
        slide_file_path = os.path.join(data_slide_dir, f"{wsi_type}", f"{slide_id}.tiff")
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=img_transforms)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        compute_w_loader(output_h5_path, output_pt_path, loader, model, 1)

    return output_h5_path, output_pt_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--coords_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--patients_csv', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
    current_time = datetime.now()
    print("Start time：", current_time)
    patients_df = pd.read_csv(args.patients_csv)
    wsi_types = ["CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3", "PDL1", "HE"]
    
    os.makedirs(args.feat_dir, exist_ok=True)

    for _, patient_row in tqdm(patients_df.iterrows(), total=patients_df.shape[0]):
        patient_id = patient_row['Patient_ID']
        extract_patient_features(patient_id, wsi_types, args.data_slide_dir, args.coords_dir, args.feat_dir, args.batch_size, 
                                 args.model_name, args.target_patch_size)
    current_time = datetime.now()
    print("End time：", current_time)