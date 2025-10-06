/**
 * Frontend Validation Utilities
 * Mirrors backend validation logic from src/config_loaders/validate_strategy_config.py
 *
 * Returns validation results with errors, warnings, and info messages
 */

/**
 * Validate client configuration (Priority 1: Critical)
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
    strict_mode = "true"
  } = config;

  // Rule 1: Min client bounds - all min_* values must be <= num_of_clients
  if (min_fit_clients > num_of_clients) {
    errors.push({
      field: "min_fit_clients",
      message: `Cannot require more fit clients (${min_fit_clients}) than total clients (${num_of_clients})`
    });
  }

  if (min_evaluate_clients > num_of_clients) {
    errors.push({
      field: "min_evaluate_clients",
      message: `Cannot require more evaluate clients (${min_evaluate_clients}) than total clients (${num_of_clients})`
    });
  }

  if (min_available_clients > num_of_clients) {
    errors.push({
      field: "min_available_clients",
      message: `Cannot require more available clients (${min_available_clients}) than total clients (${num_of_clients})`
    });
  }

  // Rule 2: Strict mode warnings - backend auto-corrects these values
  if (strict_mode === "true") {
    if (min_fit_clients !== num_of_clients && min_fit_clients <= num_of_clients) {
      warnings.push({
        field: "min_fit_clients",
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`
      });
    }
    if (min_evaluate_clients !== num_of_clients && min_evaluate_clients <= num_of_clients) {
      warnings.push({
        field: "min_evaluate_clients",
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`
      });
    }
    if (min_available_clients !== num_of_clients && min_available_clients <= num_of_clients) {
      warnings.push({
        field: "min_available_clients",
        message: `Strict mode will set this to ${num_of_clients} (all clients participate)`
      });
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate strategy-specific parameters (Priority 2)
 * Source: validate_strategy_config.py:124-157
 */
export function validateStrategyParams(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { aggregation_strategy_keyword } = config;

  // Trust strategy
  if (aggregation_strategy_keyword === "trust") {
    const requiredParams = [
      { key: "begin_removing_from_round", label: "Begin Removing From Round" },
      { key: "trust_threshold", label: "Trust Threshold" },
      { key: "beta_value", label: "Beta Value" },
      { key: "num_of_clusters", label: "Number of Clusters" }
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for trust aggregation strategy`
        });
      }
    });
  }

  // PID strategies
  if (["pid", "pid_scaled", "pid_standardized"].includes(aggregation_strategy_keyword)) {
    const requiredParams = [
      { key: "num_std_dev", label: "Number of Std Deviations" },
      { key: "Kp", label: "Kp (Proportional Gain)" },
      { key: "Ki", label: "Ki (Integral Gain)" },
      { key: "Kd", label: "Kd (Derivative Gain)" }
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for ${aggregation_strategy_keyword} strategy`
        });
      }
    });
  }

  // Krum-based strategies
  if (["multi-krum", "krum", "multi-krum-based"].includes(aggregation_strategy_keyword)) {
    if (config.num_krum_selections === undefined || config.num_krum_selections === null || config.num_krum_selections === '') {
      errors.push({
        field: "num_krum_selections",
        message: `Krum Selections is required for ${aggregation_strategy_keyword} strategy`
      });
    }
  }

  // Trimmed mean strategy
  if (aggregation_strategy_keyword === "trimmed_mean") {
    if (config.trim_ratio === undefined || config.trim_ratio === null || config.trim_ratio === '') {
      errors.push({
        field: "trim_ratio",
        message: "Trim Ratio is required for trimmed mean strategy"
      });
    } else if (config.trim_ratio <= 0 || config.trim_ratio >= 0.5) {
      errors.push({
        field: "trim_ratio",
        message: "Trim Ratio must be between 0 and 0.5 (exclusive)"
      });
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate attack configuration (Priority 3)
 * Source: validate_strategy_config.py:159-171
 */
export function validateAttackConfig(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { attack_type, num_of_malicious_clients } = config;

  // Only validate attack params if there are malicious clients
  if (num_of_malicious_clients > 0) {
    if (attack_type === "gaussian_noise") {
      const requiredParams = [
        { key: "gaussian_noise_mean", label: "Gaussian Noise Mean" },
        { key: "gaussian_noise_std", label: "Gaussian Noise Std" },
        { key: "attack_ratio", label: "Attack Ratio" }
      ];

      requiredParams.forEach(({ key, label }) => {
        if (config[key] === undefined || config[key] === null || config[key] === '') {
          errors.push({
            field: key,
            message: `${label} is required for gaussian_noise attack`
          });
        }
      });
    }
  }

  return { errors, warnings, infos };
}

/**
 * Validate LLM configuration (Priority 4)
 * Source: validate_strategy_config.py:174-205
 */
export function validateLLMConfig(config) {
  const errors = [];
  const warnings = [];
  const infos = [];

  const { use_llm, model_type, llm_task, llm_finetuning } = config;

  if (use_llm === "true") {
    // LLM only supported for transformer models
    if (model_type !== "transformer") {
      errors.push({
        field: "use_llm",
        message: "LLM finetuning is only supported for transformer models"
      });
    }

    // Required LLM parameters
    const requiredParams = [
      { key: "llm_model", label: "LLM Model" },
      { key: "llm_finetuning", label: "LLM Finetuning" },
      { key: "llm_task", label: "LLM Task" },
      { key: "llm_chunk_size", label: "LLM Chunk Size" }
    ];

    requiredParams.forEach(({ key, label }) => {
      if (config[key] === undefined || config[key] === null || config[key] === '') {
        errors.push({
          field: key,
          message: `${label} is required for LLM finetuning`
        });
      }
    });

    // MLM task-specific parameters
    if (llm_task === "mlm") {
      if (config.mlm_probability === undefined || config.mlm_probability === null || config.mlm_probability === '') {
        errors.push({
          field: "mlm_probability",
          message: "MLM Probability is required for MLM task"
        });
      }
    }

    // LoRA finetuning-specific parameters
    if (llm_finetuning === "lora") {
      const loraParams = [
        { key: "lora_rank", label: "LoRA Rank" },
        { key: "lora_alpha", label: "LoRA Alpha" },
        { key: "lora_dropout", label: "LoRA Dropout" },
        { key: "lora_target_modules", label: "LoRA Target Modules" }
      ];

      loraParams.forEach(({ key, label }) => {
        if (config[key] === undefined || config[key] === null || config[key] === '' || (Array.isArray(config[key]) && config[key].length === 0)) {
          errors.push({
            field: key,
            message: `${label} is required for LoRA finetuning`
          });
        }
      });
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
  const clientValidation = validateClientConfig(config);
  const strategyValidation = validateStrategyParams(config);
  const attackValidation = validateAttackConfig(config);
  const llmValidation = validateLLMConfig(config);

  // Aggregate all errors, warnings, and infos
  const errors = [
    ...clientValidation.errors,
    ...strategyValidation.errors,
    ...attackValidation.errors,
    ...llmValidation.errors
  ];

  const warnings = [
    ...clientValidation.warnings,
    ...strategyValidation.warnings,
    ...attackValidation.warnings,
    ...llmValidation.warnings
  ];

  const infos = [
    ...clientValidation.infos,
    ...strategyValidation.infos,
    ...attackValidation.infos,
    ...llmValidation.infos
  ];

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    infos
  };
}
