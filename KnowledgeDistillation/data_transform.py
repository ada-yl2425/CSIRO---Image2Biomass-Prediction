from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide

def transForms(args, num_bins, mean, std, p, type):
    if type == 'train':
        train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        TrivialAugmentWide(
            num_magnitude_bins=num_bins,  # 更多强度级别
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=None
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p),
        ])
        return train_transforms


    elif type == 'val':
        val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
        return val_transforms

