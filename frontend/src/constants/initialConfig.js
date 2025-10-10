// Default configuration values from config/simulation_strategies/example_strategy_config.json
export const initialConfig = {
  // Simulation Naming
  display_name: '',

  // Core Simulation Settings
  num_of_rounds: 4,
  num_of_clients: 5,
  num_of_malicious_clients: 1,
  attack_ratio: 1.0,

  // Dataset & Model
  dataset_source: 'local', // "local" | "huggingface"
  dataset_keyword: 'femnist_iid',
  hf_dataset_name: '', // HuggingFace dataset name
  partitioning_strategy: 'iid', // "iid" | "dirichlet" | "pathological"
  partitioning_params: {}, // e.g., {"alpha": 0.5} for Dirichlet
  model_type: 'cnn',
  use_llm: 'false',

  // Attack Configuration
  attack_type: 'gaussian_noise',
  gaussian_noise_mean: 0,
  gaussian_noise_std: 75,
  num_std_dev: 2,

  // Dynamic Poisoning Attacks
  dynamic_attacks: {
    enabled: false,
    schedule: [],
  },

  // Defense Strategy
  aggregation_strategy_keyword: 'pid',
  remove_clients: 'true',
  begin_removing_from_round: 2,

  // Trust & Reputation
  trust_threshold: 0.15,
  beta_value: 0.75,

  // PID Parameters
  Kp: 1,
  Ki: 0.05,
  Kd: 0.05,

  // Krum Settings
  num_krum_selections: 3,
  num_of_clusters: 1,

  // Training Configuration
  training_device: 'cpu',
  cpus_per_client: 1,
  gpus_per_client: 0.0,
  num_of_client_epochs: 1,
  batch_size: 20,
  training_subset_fraction: 0.9,

  // Client Requirements
  min_fit_clients: 5,
  min_evaluate_clients: 5,
  min_available_clients: 5,
  evaluate_metrics_aggregation_fn: 'weighted_average',

  // Output Settings
  show_plots: 'false',
  save_plots: 'true',
  save_csv: 'true',
  preserve_dataset: 'true',
  strict_mode: 'true',

  // LLM Settings
  llm_model: 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
  llm_finetuning: 'lora',
  llm_task: 'mlm',
  llm_chunk_size: 256,
  mlm_probability: 0.15,
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.1,
  lora_target_modules: ['query', 'value'],
};
