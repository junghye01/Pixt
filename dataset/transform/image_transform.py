from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToTensor


class Pixt_ImageTransform:
    def __init__(self):
        self.transform = Compose(
            [
                Resize((256, 256)),
                CenterCrop((224, 224)),
                RandomHorizontalFlip(0.5),
                ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.transform(x)
