from torchvision.datasets import CelebA

class MyCelebA(CelebA):
    def __init__(self, root, split="train", transform=None,
                 target_transform=None, download=False):
        super(MyCelebA, self).__init__(root, split, "attr", transform, target_transform, download)

    def __getitem__(self, item):
        img, label = super(MyCelebA, self).__getitem__(item)
        code = label[20]
        return img, code