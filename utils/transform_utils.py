from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms


def get_transforms_with_randaugment(mean, std, target_img_size, n=2, m=9):
    trsforms = []
    
    trsforms.append(transforms.Resize((target_img_size, target_img_size)))
    trsforms.append(transforms.RandAugment(num_ops=n, magnitude=m))
    trsforms.append(transforms.ToTensor())
    trsforms.append(transforms.Normalize(mean, std))

    trsforms = transforms.Compose(trsforms)
    return trsforms