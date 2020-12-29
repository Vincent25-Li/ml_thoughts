import numpy as np

from torchvision import datasets

def get_mean_std():
    dataset = datasets.CIFAR10('./data', train=True, download=True)
    image_mean_accu = np.zeros(3)
    image_std_accu = np.zeros(3)
    total = 0
    for data in dataset:
        image = np.asarray(data[0]).reshape(-1, 3).astype(np.int)

        image_mean_accu += image.mean(axis=0)
        image_std_accu += np.square(image).mean(axis=0)
        total += 1

    image_means = image_mean_accu / total
    image_stds = np.sqrt(image_std_accu / total - np.square(image_means))
    return image_means, image_stds

image_means, image_stds = get_mean_std()