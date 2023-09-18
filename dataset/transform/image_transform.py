from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize,ToTensor


class Pixt_ImageTransform:
    def __init__(self):
        self.transform = Compose(
            [
                Resize((256, 256)),
                CenterCrop((224, 224)),
                RandomHorizontalFlip(0.5),
                RandomRotation(degrees=(-15,15)),
                ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                ToTensor(),
                
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class Pixt_ValidationTest_Transform:
    def __init__(self):
        self.transform=Compose(
            [
                Resize((256,256)),
                CenterCrop((224,224)),
                ToTensor(),
                
            ]
        )

    def __call__(self,x):
        return self.transform(x)
