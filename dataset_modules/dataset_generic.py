import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].patient_data['Patient_ID'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])
	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path,
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		ignore=[],
		label_col = None,
		):

		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.train_ids, self.val_ids  = (None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		
		patient_data = pd.read_csv(csv_path, dtype=str)

		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(patient_data)

		self.patient_data = patient_data
		
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):

		# store ids corresponding each class at the slide level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == str(i))[0]


	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data


	def __len__(self):
			return len(self.patient_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.patient_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), label_frac = 1.0, method='montecarlo'):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'label_frac': label_frac,
					'seed': self.seed,
					'method': method
					}

		settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)
			
		self.train_ids, self.val_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.patient_data['Patient_ID'].isin(split.tolist())
			df_slice = self.patient_data[mask].reset_index(drop=True)
			print("len(df_slice): ", len(df_slice))
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, use_h5=self.use_h5, wsi_types = self.wsi_types)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):

		if from_id:

			if len(self.train_ids) > 0:
				train_data = self.patient_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir = self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.patient_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir = self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=str)
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			
		return train_split, val_split


	def getlabel(self, ids):
		return self.patient_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0

		if return_descriptor:
			return df


class Generic_features_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, use_h5=None, wsi_types = ['HE'],
		**kwargs):
	
		super(Generic_features_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = use_h5
		self.wsi_types = wsi_types

	def __getitem__(self, idx):
		patient_id = self.patient_data['Patient_ID'][idx]
		label = self.patient_data['label'][idx]
		data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				all_features = []
				for wsi_type in self.wsi_types:
					full_path = os.path.join(data_dir, wsi_type, 'pt_files', '{}.pt'.format(patient_id))
					features = torch.load(full_path)
					all_features.append(features)
				features = torch.stack(all_features, dim=0)
				return features, label
			
			else:
				return patient_id, label
			
		# else:
		# 	all_features = []
		# 	for wsi_type in self.wsi_types:
		# 		full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(patient_id))
		# 		with h5py.File(full_path,'r') as hdf5_file:
		# 			feature = hdf5_file['features'][:]
		# 			coords = hdf5_file['coords'][:]

		# 		feature = torch.from_numpy(feature)
		# 		all_features.append(feature)
		# 	features = torch.stack(all_features, dim=0)
			
		# 	return features, label
		

	def return_splits(self, from_id=True, csv_path=None):

		if from_id:

			if len(self.train_ids) > 0:
				train_data = self.patient_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir = self.data_dir, num_classes=self.num_classes, use_h5=self.use_h5, wsi_types = self.wsi_types)
			else:
				train_split = None

			if len(self.val_ids) > 0:
				val_data = self.patient_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir = self.data_dir, num_classes=self.num_classes, use_h5=self.use_h5, wsi_types = self.wsi_types)
			else:
				val_split = None
			
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=str)
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			
		return train_split, val_split



class Generic_Split(Generic_features_Dataset):
	def __init__(self, patient_data, data_dir, num_classes=2, use_h5 = False, wsi_types = ['CD4']):

		self.use_h5 = use_h5
		self.wsi_types = wsi_types
		self.patient_data = patient_data
		self.num_classes = num_classes
		self.data_dir = data_dir
		self.cls_ids_prep()
		