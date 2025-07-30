from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
emnist = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)