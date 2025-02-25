from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, X_data, y_data, split, transform=None):
        self.X_data = X_data
        self.y_data = y_data
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):
        image = self.X_data[idx]
        depth = self.y_data[idx]
        
        if self.split == 'train':
            depth = depth /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001
        
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.X_data)