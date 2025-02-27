from datasets.nyu_dataset import NYUDataset


class GuideDepthDataset(NYUDataset):

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        depth = sample["depth"]

        if not self.val:
            depth = depth / 255.0 * 10.0  #From 8bit to range [0, 10] (meter)
        else:
            depth = depth * 0.001

        sample.update("depth", depth)

        if self.transform:
            sample = self.transform(sample)

        return sample
