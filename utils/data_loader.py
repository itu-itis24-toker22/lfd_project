import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VisADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, class_name=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # The CSV file defines the train/test split
        csv_path = os.path.join(self.root_dir, '1cls.csv')
        data_df = pd.read_csv(csv_path)

        # Filter for a specific class if provided
        if class_name:
            print(f"Filtering dataset for class: {class_name}")
            data_df = data_df[data_df['object'] == class_name]

        # Filter for train or test set
        split_df = data_df[data_df['split'] == split]

        for idx, row in split_df.iterrows():
            img_path = os.path.join(self.root_dir, 'VisA_20220922', row['image'])
            
            self.image_paths.append(img_path)
            self.labels.append(1 if row['label'] == 'anomaly' else 0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loader(data_path, batch_size=32, image_size=256, is_train=True, class_name=None):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    split = 'train' if is_train else 'test'
    dataset = VisADataset(root_dir=data_path, split=split, transform=transform, class_name=class_name)

    # Set num_workers to 0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    return dataloader
