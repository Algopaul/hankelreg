import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hankelreg.config import DatasetConfig


def get_dataset(cfg: DatasetConfig):
  torch_data_dir = os.path.join(str(os.getenv('TMPDIR')), 'data')
  transform = transforms.Compose([transforms.ToTensor()])
  train_ds = datasets.MNIST(
      torch_data_dir,
      train=True,
      download=True,
      transform=transform,
  )
  test_ds = datasets.MNIST(
      torch_data_dir, train=False, download=True, transform=transform)
  train_loader = DataLoader(
      train_ds,
      batch_size=cfg.batch_dim,
      shuffle=True,
      prefetch_factor=3,
      num_workers=1)
  test_loader = DataLoader(
      test_ds,
      batch_size=cfg.batch_dim,
      shuffle=True,
      prefetch_factor=3,
      num_workers=1)
  return train_loader, test_loader
