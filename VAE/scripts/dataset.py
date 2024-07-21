from torchvision import transforms

# Load Data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0), (1,)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
    ]
)
