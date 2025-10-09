/**
 * Frontend Validation Utilities
 * Mirrors backend validation logic from src/config_loaders/validate_strategy_config.py
 *
 * Returns validation results with errors, warnings, and info messages
 */

/**
 * Validate basic numeric parameters
 */
export function validateBasicParams(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const {
    num_of_rounds,
    num_of_clients,
    batch_size,
    num_of_client_epochs,
    training_subset_fraction,
  } = config;

  // num_of_rounds validation
  if (num_of_rounds === undefined || num_of_rounds === null || num_of_rounds === '') {
    errors.push({
      field: 'num_of_rounds',
      message: 'Number of rounds is required',
    });
  } else if (num_of_rounds <= 0) {
    errors.push({
      field: 'num_of_rounds',
      message: 'Number of rounds must be greater than 0',
    });
  } else if (num_of_rounds < 3) {
    warnings.push({
      field: 'num_of_rounds',
      message:
        'Very few rounds may not show meaningful convergence. Consider at least 5-10 rounds.',
    });
  }

  // num_of_clients validation
  if (num_of_clients === undefined || num_of_clients === null || num_of_clients === '') {
    errors.push({
      field: 'num_of_clients',
      message: 'Number of clients is required',
    });
  } else if (num_of_clients <= 0) {
    errors.push({
      field: 'num_of_clients',
      message: 'Number of clients must be greater than 0',
    });
  } else if (num_of_clients > 50) {
    warnings.push({
      field: 'num_of_clients',
      message:
        'Large client counts (>50) may significantly slow down simulation. Consider using fewer clients for faster testing.',
    });
  }

  // batch_size validation
  if (batch_size === undefined || batch_size === null || batch_size === '') {
    errors.push({
      field: 'batch_size',
      message: 'Batch size is required',
    });
  } else if (batch_size <= 0) {
    errors.push({
      field: 'batch_size',
      message: 'Batch size must be greater than 0',
    });
  } else if (batch_size < 10) {
    warnings.push({
      field: 'batch_size',
      message:
        'Very small batch sizes (<10) may cause training instability and slower convergence.',
    });
  }

  // num_of_client_epochs validation
  if (
    num_of_client_epochs === undefined ||
    num_of_client_epochs === null ||
    num_of_client_epochs === ''
  ) {
    errors.push({
      field: 'num_of_client_epochs',
      message: 'Client epochs is required',
    });
  } else if (num_of_client_epochs <= 0) {
    errors.push({
      field: 'num_of_client_epochs',
      message: 'Client epochs must be greater than 0',
    });
  }

  // training_subset_fraction validation
  if (
    training_subset_fraction === undefined ||
    training_subset_fraction === null ||
    training_subset_fraction === ''
  ) {
    errors.push({
      field: 'training_subset_fraction',
      message: 'Training subset fraction is required',
    });
  } else if (training_subset_fraction <= 0 || training_subset_fraction > 1) {
    errors.push({
      field: 'training_subset_fraction',
      message: 'Training subset fraction must be between 0 and 1 (exclusive of 0, inclusive of 1)',
    });
  }

  return { errors, warnings, infos };
}

/**
 * Validate client configuration
 * Source: validate_strategy_config.py:207-253
 */
export function validateClientConfig(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const {
    num_of_clients,
    min_fit_clients,
    min_evaluate_clients,
    min_available_clients,
    strict_mode = 'true',
  } = config;

  // Min client bounds - all min_* values must be <= num_of_clients
  if (min_fit_clients > num_of_clients) {
    errors.push({
      field: 'min_fit_clients',
      message: `Cannot require more fit clients (${min_fit_clients}) than total clients (${num_of_clients})`,
    });
  }

  if (min_evaluate_clients > num_of_clients) {
    errors.push({
      field: 'min_evaluate_clients',
      message: `Cannot require more evaluate clients (${min_evaluate_clients}) than total clients (${num_of_clients})`,
    });
  }

  if (min_available_clients > num_of_clients) {
    errors.push({
      field: 'min_available_clients',
      message: `Cannot require more available clients (${min_available_clients}) than total clients (${num_of_clients})`,
    });
  }

  // Strict mode warnings
  if (strict_mode === 'true') {
    if (min_fit_clients !== num_of_clients && min_fit_clients <= num_of_clients) {
      warnings.push({
        field: 'min_fit_clients',
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`,
      });
    }
    if (min_evaluate_clients !== num_of_clients && min_evaluate_clients <= num_of_clients) {
      warnings.push({
        field: 'min_evaluate_clients',
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`,
      });
    }
    if (min_available_clients !== num_of_clients && min_available_clients <= num_of_clients) {
      warnings.push({
        field: 'min_available_clients',
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`,
      });
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate strategy-specific parameters
 * Source: validate_strategy_config.py:124-157
 */
export function validateStrategyParams(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { aggregation_strategy_keyword } = config;

  // Trust strategy
  if (aggregation_strategy_keyword === 'trust') {
    const requiredParams = [
      { key: 'begin_removing_from_round', label: 'Begin Removing From Round' },
      { key: 'trust_threshold', label: 'Trust Threshold' },
      { key: 'beta_value', label: 'Beta Value' },
      { key: 'num_of_clusters', label: 'Number of Clusters' },
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for trust aggregation strategy`,
        });
      }
    });
  }

  // PID strategies
  if (['pid', 'pid_scaled', 'pid_standardized'].includes(aggregation_strategy_keyword)) {
    const requiredParams = [
      { key: 'num_std_dev', label: 'Number of Std Deviations' },
      { key: 'Kp', label: 'Kp (Proportional Gain)' },
      { key: 'Ki', label: 'Ki (Integral Gain)' },
      { key: 'Kd', label: 'Kd (Derivative Gain)' },
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for ${aggregation_strategy_keyword} strategy`,
        });
      }
    });
  }

  // Krum-based strategies
  if (['multi-krum', 'krum', 'multi-krum-based'].includes(aggregation_strategy_keyword)) {
    if (
      config.num_krum_selections === undefined ||
      config.num_krum_selections === null ||
      config.num_krum_selections === ''
    ) {
      errors.push({
        field: 'num_krum_selections',
        message: `Krum Selections is required for ${aggregation_strategy_keyword} strategy`,
      });
    } else {
      const { num_krum_selections, num_of_clients, num_of_malicious_clients = 0 } = config;

      // Error: num_krum_selections must be < num_of_clients
      if (num_krum_selections >= num_of_clients) {
        errors.push({
          field: 'num_krum_selections',
          message: `Krum Selections (${num_krum_selections}) must be less than total clients (${num_of_clients})`,
        });
      }

      // Warning: Recommend optimal value
      const recommended = num_of_clients - num_of_malicious_clients - 2;
      if (recommended > 0 && num_krum_selections > recommended) {
        warnings.push({
          field: 'num_krum_selections',
          message: `For best Byzantine robustness with ${num_of_malicious_clients} malicious clients, consider setting to ${recommended} or lower`,
        });
      }
    }
  }

  // Bulyan strategy - special validation
  if (aggregation_strategy_keyword === 'bulyan') {
    if (
      config.num_krum_selections === undefined ||
      config.num_krum_selections === null ||
      config.num_krum_selections === ''
    ) {
      errors.push({
        field: 'num_krum_selections',
        message: 'Krum Selections is required for bulyan strategy',
      });
    } else {
      const { num_krum_selections, num_of_clients } = config;

      // Bulyan constraint: (n - C) must be even for n - C = 2f
      const diff = num_of_clients - num_krum_selections;
      if (diff % 2 !== 0) {
        errors.push({
          field: 'num_krum_selections',
          message: `Bulyan requires (clients - selections) to be even. With ${num_of_clients} clients, try ${num_krum_selections - 1} or ${num_krum_selections + 1} selections`,
        });
      }

      if (num_krum_selections >= num_of_clients) {
        errors.push({
          field: 'num_krum_selections',
          message: `Krum Selections (${num_krum_selections}) must be less than total clients (${num_of_clients})`,
        });
      }
    }
  }

  // Trimmed mean strategy
  if (aggregation_strategy_keyword === 'trimmed_mean') {
    if (config.trim_ratio === undefined || config.trim_ratio === null || config.trim_ratio === '') {
      errors.push({
        field: 'trim_ratio',
        message: 'Trim Ratio is required for trimmed mean strategy',
      });
    } else if (config.trim_ratio <= 0 || config.trim_ratio >= 0.5) {
      errors.push({
        field: 'trim_ratio',
        message: 'Trim Ratio must be between 0 and 0.5 (exclusive)',
      });
    }
  }

  // FedAvg strategy - no per-client plots
  if (aggregation_strategy_keyword === 'fedavg') {
    infos.push({
      field: 'aggregation_strategy_keyword',
      message:
        'FedAvg only produces round-level plots (loss/accuracy convergence). For per-client visualizations, try Krum, Multi-Krum, or PID strategies.',
    });
  }

  return { errors, warnings, infos };
}

/**
 * Validate attack configuration
 * Source: validate_strategy_config.py:159-171
 */
export function validateAttackConfig(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { attack_type, num_of_malicious_clients, num_of_clients } = config;

  // Validate malicious client count
  if (num_of_malicious_clients < 0) {
    errors.push({
      field: 'num_of_malicious_clients',
      message: 'Number of malicious clients cannot be negative',
    });
  }

  if (num_of_malicious_clients > num_of_clients) {
    errors.push({
      field: 'num_of_malicious_clients',
      message: `Cannot have more malicious clients (${num_of_malicious_clients}) than total clients (${num_of_clients})`,
    });
  }

  // Only validate attack params if there are malicious clients
  if (num_of_malicious_clients > 0) {
    if (attack_type === 'gaussian_noise') {
      const requiredParams = [
        { key: 'gaussian_noise_mean', label: 'Gaussian Noise Mean' },
        { key: 'gaussian_noise_std', label: 'Gaussian Noise Std' },
        { key: 'attack_ratio', label: 'Attack Ratio' },
      ];

      requiredParams.forEach(({ key, label }) => {
        if (config[key] === undefined || config[key] === null || config[key] === '') {
          errors.push({
            field: key,
            message: `${label} is required for gaussian_noise attack`,
          });
        }
      });

      // Validate gaussian_noise_std >= 0
      if (
        config.gaussian_noise_std !== undefined &&
        config.gaussian_noise_std !== null &&
        config.gaussian_noise_std !== ''
      ) {
        if (config.gaussian_noise_std < 0) {
          errors.push({
            field: 'gaussian_noise_std',
            message: 'Gaussian Noise Std cannot be negative',
          });
        }
      }

      // Validate attack_ratio between 0 and 1
      if (
        config.attack_ratio !== undefined &&
        config.attack_ratio !== null &&
        config.attack_ratio !== ''
      ) {
        if (config.attack_ratio <= 0 || config.attack_ratio > 1) {
          errors.push({
            field: 'attack_ratio',
            message: 'Attack Ratio must be between 0 and 1 (exclusive of 0, inclusive of 1)',
          });
        }
      }
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate dataset and model type compatibility
 */
export function validateDatasetModelCompatibility(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { model_type, dataset_source, dataset_keyword, hf_dataset_name } = config;

  // Define dataset modalities
  const IMAGE_DATASETS_LOCAL = [
    'femnist_iid',
    'femnist_niid',
    'its',
    'pneumoniamnist',
    'bloodmnist',
    'lung_photos',
  ];
  const TEXT_DATASETS_LOCAL = ['flair', 'medquad'];

  // Common HuggingFace dataset patterns
  const IMAGE_DATASET_PATTERNS = [
    'mnist',
    'cifar',
    'femnist',
    'fashion',
    'svhn',
    'imagenet',
    'coco',
    'celeb',
    'flowers',
    'food',
    'medmnist',
    'pneumonia',
    'chest',
    'xray',
    'blood',
    'skin',
  ];

  const TEXT_DATASET_PATTERNS = [
    'imdb',
    'shakespeare',
    'ag_news',
    'yelp',
    'sst',
    'glue',
    'squad',
    'wikitext',
    'bookcorpus',
    'amazon',
    'tweet',
    'sentiment',
    'review',
    'qa',
    'nli',
    'flair',
    'medquad',
    'pubmed',
    'mimic',
  ];

  let datasetModality = null;

  // Determine dataset modality
  if (dataset_source === 'local') {
    if (IMAGE_DATASETS_LOCAL.includes(dataset_keyword)) {
      datasetModality = 'image';
    } else if (TEXT_DATASETS_LOCAL.includes(dataset_keyword)) {
      datasetModality = 'text';
    }
  } else if (dataset_source === 'huggingface' && hf_dataset_name) {
    const nameLower = hf_dataset_name.toLowerCase();

    // Check if it matches known image patterns
    const isImage = IMAGE_DATASET_PATTERNS.some(pattern => nameLower.includes(pattern));
    const isText = TEXT_DATASET_PATTERNS.some(pattern => nameLower.includes(pattern));

    if (isImage && !isText) {
      datasetModality = 'image';
    } else if (isText && !isImage) {
      datasetModality = 'text';
    } else if (isImage && isText) {
      // Ambiguous - could be multimodal
      datasetModality = 'unknown';
    }
  }

  // Validate model_type vs dataset modality
  if (datasetModality && model_type) {
    if (model_type === 'cnn' && datasetModality === 'text') {
      errors.push({
        field: 'model_type',
        message: `CNN models don't work with text datasets. Please select "Transformer" model type or choose an image dataset.`,
      });
    } else if (model_type === 'transformer' && datasetModality === 'image') {
      warnings.push({
        field: 'model_type',
        message: `Transformer models are slower for image datasets. Consider using CNN for better performance, or keep Transformer if you need LLM finetuning.`,
      });
    } else if (datasetModality === 'unknown') {
      infos.push({
        field: 'hf_dataset_name',
        message:
          'Unable to auto-detect dataset type. Ensure your model type (CNN for images, Transformer for text) matches your dataset.',
      });
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate LLM configuration
 * Source: validate_strategy_config.py:174-205
 */
export function validateLLMConfig(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { use_llm, model_type, llm_task, llm_finetuning } = config;

  if (use_llm === 'true') {
    // LLM only supported for transformer models
    if (model_type !== 'transformer') {
      errors.push({
        field: 'use_llm',
        message: 'LLM finetuning is only supported for transformer models',
      });
    }

    // Required LLM parameters
    const requiredParams = [
      { key: 'llm_model', label: 'LLM Model' },
      { key: 'llm_finetuning', label: 'LLM Finetuning' },
      { key: 'llm_task', label: 'LLM Task' },
      { key: 'llm_chunk_size', label: 'LLM Chunk Size' },
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for LLM finetuning`,
        });
      }
    });

    // MLM task-specific parameters
    if (llm_task === 'mlm') {
      if (
        config.mlm_probability === undefined ||
        config.mlm_probability === null ||
        config.mlm_probability === ''
      ) {
        errors.push({
          field: 'mlm_probability',
          message: 'MLM Probability is required for MLM task',
        });
      }
    }

    // LoRA finetuning-specific parameters
    if (llm_finetuning === 'lora') {
      const loraParams = [
        { key: 'lora_rank', label: 'LoRA Rank' },
        { key: 'lora_alpha', label: 'LoRA Alpha' },
        { key: 'lora_dropout', label: 'LoRA Dropout' },
        { key: 'lora_target_modules', label: 'LoRA Target Modules' },
      ];

      loraParams.forEach(({ key, label }) => {
        if (
          config[key] === undefined ||
          config[key] === null ||
          config[key] === '' ||
          (Array.isArray(config[key]) && config[key].length === 0)
        ) {
          errors.push({
            field: key,
            message: `${label} is required for LoRA finetuning`,
          });
        }
      });

      // Validate lora_rank > 0
      if (config.lora_rank !== undefined && config.lora_rank !== null && config.lora_rank !== '') {
        if (config.lora_rank <= 0) {
          errors.push({
            field: 'lora_rank',
            message: 'LoRA Rank must be greater than 0',
          });
        }
      }

      // Validate lora_alpha > 0
      if (
        config.lora_alpha !== undefined &&
        config.lora_alpha !== null &&
        config.lora_alpha !== ''
      ) {
        if (config.lora_alpha <= 0) {
          errors.push({
            field: 'lora_alpha',
            message: 'LoRA Alpha must be greater than 0',
          });
        }
      }

      // Validate lora_dropout between 0 and 1
      if (
        config.lora_dropout !== undefined &&
        config.lora_dropout !== null &&
        config.lora_dropout !== ''
      ) {
        if (config.lora_dropout < 0 || config.lora_dropout > 1) {
          errors.push({
            field: 'lora_dropout',
            message: 'LoRA Dropout must be between 0 and 1 (inclusive)',
          });
        }
      }
    }
  }

  return { errors, warnings, infos };
}

/**
 * Main validation function - runs all validation checks
 * Returns aggregated validation results
 */
export function validateConfig(config) {
  // Run all validation checks
  const basicValidation = validateBasicParams(config);
  const clientValidation = validateClientConfig(config);
  const strategyValidation = validateStrategyParams(config);
  const attackValidation = validateAttackConfig(config);
  const datasetModelValidation = validateDatasetModelCompatibility(config);
  const llmValidation = validateLLMConfig(config);

  // Aggregate all errors, warnings, and infos
  const errors = [
    ...basicValidation.errors,
    ...clientValidation.errors,
    ...strategyValidation.errors,
    ...attackValidation.errors,
    ...datasetModelValidation.errors,
    ...llmValidation.errors,
  ];

  const warnings = [
    ...basicValidation.warnings,
    ...clientValidation.warnings,
    ...strategyValidation.warnings,
    ...attackValidation.warnings,
    ...datasetModelValidation.warnings,
    ...llmValidation.warnings,
  ];

  const infos = [
    ...basicValidation.infos,
    ...clientValidation.infos,
    ...strategyValidation.infos,
    ...attackValidation.infos,
    ...datasetModelValidation.infos,
    ...llmValidation.infos,
  ];

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    infos,
  };
}
