/**
 * Dataset Constants
 */

export const DATASETS = [
  'femnist_iid',
  'femnist_niid',
  'its',
  'pneumoniamnist',
  'flair',
  'bloodmnist',
  'medquad',
  'lung_photos',
];

export const POPULAR_DATASETS = [
  // Image Classification - Basic
  { value: 'ylecun/mnist', label: 'MNIST - Handwritten digits (70k)' },
  { value: 'mnist', label: 'MNIST - Alternative (70k)' },
  { value: 'fashion_mnist', label: 'Fashion-MNIST - Clothing items (70k)' },

  // Image Classification - Standard
  { value: 'uoft-cs/cifar10', label: 'CIFAR-10 - 32x32 RGB, 10 classes (60k)' },
  { value: 'cifar10', label: 'CIFAR-10 - Alternative (60k)' },
  { value: 'uoft-cs/cifar100', label: 'CIFAR-100 - 100 classes (60k)' },
  { value: 'cifar100', label: 'CIFAR-100 - Alternative (60k)' },

  // Federated Datasets
  { value: 'flwrlabs/femnist', label: 'FEMNIST - Federated handwriting (814k)' },
  { value: 'flwrlabs/shakespeare', label: 'Shakespeare - Federated text (4.2M)' },

  // Text Classification
  { value: 'imdb', label: 'IMDB - Movie reviews sentiment (100k)' },
];
