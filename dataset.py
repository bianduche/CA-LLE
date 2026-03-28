# CA-LLE: paired / unpaired low-light datasets
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EnhancedLowLightDataset(Dataset):
    """Low-light dataset with optional paired ground-truth images."""

    def __init__(self, low_dir, gt_dir=None, size=256, is_train=True):
        self.low_dir = Path(low_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.size = size
        self.is_train = is_train

        self.low_files = sorted(
            list(self.low_dir.glob('*.png')) + list(self.low_dir.glob('*.jpg'))
        )

        if self.gt_dir:
            self.paired = []
            gt_files = sorted(
                list(self.gt_dir.glob('*.png')) + list(self.gt_dir.glob('*.jpg'))
            )
            gt_dict = {f.stem: f for f in gt_files}

            for low_file in self.low_files:
                stem = low_file.stem
                if stem in gt_dict:
                    self.paired.append((low_file, gt_dict[stem]))
                else:
                    print(f"Warning: GT not found for {low_file.name}")
            print(f"Found {len(self.paired)} paired images")
        else:
            self.paired = None
            print(f"Found {len(self.low_files)} unpaired images")

    def __len__(self):
        return len(self.paired) if self.paired else len(self.low_files)

    def __getitem__(self, idx):
        if self.paired:
            low_path, gt_path = self.paired[idx]
            low_img = Image.open(low_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        else:
            low_path = self.low_files[idx]
            low_img = Image.open(low_path).convert('RGB')
            gt_img = None

        if self.is_train:
            if gt_img:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_img, output_size=(self.size, self.size)
                )
                low_img = transforms.functional.crop(low_img, i, j, h, w)
                gt_img = transforms.functional.crop(gt_img, i, j, h, w)

                if random.random() > 0.5:
                    low_img = transforms.functional.hflip(low_img)
                    gt_img = transforms.functional.hflip(gt_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.vflip(low_img)
                    gt_img = transforms.functional.vflip(gt_img)
            else:
                low_img = transforms.RandomCrop(self.size)(low_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.hflip(low_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.vflip(low_img)
        else:
            low_img = transforms.Resize((self.size, self.size))(low_img)
            if gt_img:
                gt_img = transforms.Resize((self.size, self.size))(gt_img)

        low_tensor = transforms.ToTensor()(low_img)
        gt_tensor = transforms.ToTensor()(gt_img) if gt_img else None

        if gt_tensor is not None:
            return low_tensor, gt_tensor, str(low_path.name)
        return low_tensor, str(low_path.name)
