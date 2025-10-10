// Human-readable formatting functions for simulation parameters

export const formatStrategyName = keyword => {
  const names = {
    fedavg: 'Federated Averaging',
    trust: 'Trust-based',
    pid: 'PID Controller',
    pid_scaled: 'PID Controller (Scaled)',
    pid_standardized: 'PID Controller (Standardized)',
    'multi-krum': 'Multi-Krum',
    krum: 'Krum',
    'multi-krum-based': 'Multi-Krum Based',
    trimmed_mean: 'Trimmed Mean',
    rfa: 'RFA',
    bulyan: 'Bulyan',
  };
  return names[keyword] || keyword;
};

export const formatDatasetName = keyword => {
  const names = {
    femnist_iid: 'FEMNIST (IID)',
    femnist_niid: 'FEMNIST (Non-IID)',
    its: 'ITS',
    pneumoniamnist: 'PneumoniaMNIST',
    flair: 'FLAIR',
    bloodmnist: 'BloodMNIST',
    medquad: 'MedQuAD',
    lung_photos: 'Lung Photos',
  };
  return names[keyword] || keyword;
};

export const formatAttackName = keyword => {
  const names = {
    gaussian_noise: 'Gaussian Noise',
    label_flipping: 'Label Flipping',
  };
  return names[keyword] || keyword;
};
