import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch
import clip


class StemDataset(Dataset):
    def __init__(self, root, mode):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.mode = mode
        self.transform = None
        # read filenames
        if self.mode == 'train':
            for i in range(15):
                filenames = glob.glob(os.path.join(root, str(i) + '_*.jpg'))
                for fn in filenames:
                    self.filenames.append((fn, i))  # (filename, label) pair
        elif self.mode == 'test':
            self.filenames = [file for file in os.listdir(root) if file.endswith('.jpg')]
            self.filenames.sort()

        self.len = len(self.filenames)
        if mode == 'train':
            print("===> Start augmenting data...")
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((240, 240)),
                transforms.CenterCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-20, 20)),
                # Resize the image into a fixed shape (height = width = 224)
                # ToTensor() should be the last one of the transforms.
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            my_model, self.preprocess = clip.load("RN50", device=device)
        else:
            # We don't need augmentations in testing and validation.
            # All we need here is to resize the PIL image and transform it into Tensor.
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        if self.mode != 'test':
            image_fn, label = self.filenames[idx]
            image = Image.open(image_fn)
            image = self.preprocess(image)
            # if self.transform is not None:
            #     image = self.transform(image)
            return image, label
        else:  # mode == test
            image_fn = self.filenames[idx]
            image = Image.open(image_fn)
            # if self.transform is not None:
            #     image = self.transform(image)
            return image, -1

    def __len__(self):
        return self.len
