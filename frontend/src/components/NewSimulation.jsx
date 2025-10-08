import { useState, useEffect, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Form,
  Button,
  Card,
  Alert,
  Accordion,
  OverlayTrigger,
  Tooltip,
  Row,
  Col,
} from 'react-bootstrap';
import { createSimulation } from '../api';
import { useConfigValidation } from '../hooks/useConfigValidation';
import { useDatasetValidation } from '../hooks/useDatasetValidation';
import ValidationSummary from './ValidationSummary';

// Defaults from config/simulation_strategies/example_strategy_config.json
const initialConfig = {
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

const STRATEGIES = [
  'fedavg',
  'trust',
  'pid',
  'pid_scaled',
  'pid_standardized',
  'multi-krum',
  'krum',
  'multi-krum-based',
  'trimmed_mean',
  'rfa',
  'bulyan',
];
const DATASETS = [
  'femnist_iid',
  'femnist_niid',
  'its',
  'pneumoniamnist',
  'flair',
  'bloodmnist',
  'medquad',
  'lung_photos',
];
const ATTACKS = ['gaussian_noise', 'label_flipping'];
const DEVICES = ['cpu', 'gpu', 'cuda'];

const POPULAR_DATASETS = [
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

const PRESETS = {
  convergence: {
    name: 'Convergence Test',
    subtitle: '15 rounds / 8 clients',
    description:
      'Tests federated learning convergence with PID defense. 15 rounds produces smooth plots.',
    estimatedTime: '60-90 seconds',
    icon: '📈',
    config: {
      num_of_rounds: 15,
      num_of_clients: 8,
      num_of_malicious_clients: 0,
      aggregation_strategy_keyword: 'pid',
      min_fit_clients: 8,
      min_evaluate_clients: 8,
      min_available_clients: 8,
    },
  },
  attack: {
    name: 'Attack Test',
    subtitle: '10 rounds / 2 malicious',
    description:
      'Tests Byzantine attack detection with 2 malicious clients using Gaussian noise. PID defense adapts threshold over 10 rounds.',
    estimatedTime: '2-3 minutes',
    icon: '⚔️',
    config: {
      num_of_rounds: 10,
      num_of_clients: 5,
      num_of_malicious_clients: 2,
      min_fit_clients: 5,
      min_evaluate_clients: 5,
      min_available_clients: 5,
    },
  },
  full: {
    name: 'Full Run',
    subtitle: '10 rounds / 10 clients',
    description:
      '10 rounds with 10 clients for detailed experiments. Use for research and benchmarking.',
    estimatedTime: '2-3 minutes',
    icon: '🔬',
    config: {
      num_of_rounds: 10,
      num_of_clients: 10,
      num_of_malicious_clients: 0,
      min_fit_clients: 10,
      min_evaluate_clients: 10,
      min_available_clients: 10,
    },
  },
};

function NewSimulation() {
  const [config, setConfig] = useState(() => {
    // Try to load saved draft from localStorage
    const savedDraft = localStorage.getItem('simulation-draft');
    if (savedDraft) {
      try {
        return JSON.parse(savedDraft);
      } catch (e) {
        console.error('Failed to parse saved draft:', e);
        return initialConfig;
      }
    }
    return initialConfig;
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [draftSaved, setDraftSaved] = useState(false);
  const navigate = useNavigate();

  // Real-time validation
  const validation = useConfigValidation(config);
  const { errors, warnings, infos } = validation;

  // HuggingFace dataset validation
  const datasetValidation = useDatasetValidation(
    config.dataset_source === 'huggingface' ? config.hf_dataset_name : null
  );

  // Helper functions to get validation messages for specific fields
  const getFieldError = fieldName => errors.find(e => e.field === fieldName);
  const getFieldWarning = fieldName => warnings.find(w => w.field === fieldName);

  // Auto-save to localStorage when config changes
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      localStorage.setItem('simulation-draft', JSON.stringify(config));
      setDraftSaved(true);
      setTimeout(() => setDraftSaved(false), 2000);
    }, 1000); // Debounce saves by 1 second

    return () => clearTimeout(timeoutId);
  }, [config]);

  const handleChange = e => {
    const { name, value, type } = e.target;
    let finalValue = value;

    if (type === 'number') {
      finalValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    }

    setConfig(prev => ({ ...prev, [name]: finalValue }));
  };

  const handlePresetChange = presetKey => {
    if (presetKey && PRESETS[presetKey]) {
      setSelectedPreset(presetKey);
      setConfig(prev => ({ ...prev, ...PRESETS[presetKey].config }));
    } else {
      setSelectedPreset(null);
    }
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const response = await createSimulation(config);
      const { simulation_id } = response.data;
      // Clear draft on successful submission
      localStorage.removeItem('simulation-draft');
      navigate(`/simulations/${simulation_id}`);
    } catch (err) {
      console.error('Failed to create simulation:', err);
      setError(err.response?.data?.detail || 'An unexpected error occurred.');
      setSubmitting(false);
    }
  };

  const handleClearDraft = () => {
    if (window.confirm('Clear saved draft and reset to defaults?')) {
      localStorage.removeItem('simulation-draft');
      setConfig(initialConfig);
      setSelectedPreset(null);
    }
  };

  const needsTrustParams = config.aggregation_strategy_keyword === 'trust';
  const needsPidParams = ['pid', 'pid_scaled', 'pid_standardized'].includes(
    config.aggregation_strategy_keyword
  );
  const needsKrumParams = ['multi-krum', 'krum', 'multi-krum-based'].includes(
    config.aggregation_strategy_keyword
  );
  const needsTrimmedMeanParams = config.aggregation_strategy_keyword === 'trimmed_mean';
  const needsGaussianParams = config.attack_type === 'gaussian_noise';
  const needsLlmParams = config.use_llm === 'true';
  const needsMLMParams = needsLlmParams && config.llm_task === 'mlm';

  const validateConfig = () => {
    if (!config.aggregation_strategy_keyword || !config.dataset_keyword) return false;
    if (config.num_of_rounds <= 0 || config.num_of_clients <= 0) return false;
    if (
      config.num_of_malicious_clients < 0 ||
      config.num_of_malicious_clients > config.num_of_clients
    )
      return false;

    if (needsTrustParams) {
      if (
        !config.begin_removing_from_round ||
        !config.trust_threshold ||
        !config.beta_value ||
        !config.num_of_clusters
      )
        return false;
    }

    if (needsPidParams) {
      if (
        config.num_std_dev === undefined ||
        config.Kp === undefined ||
        config.Ki === undefined ||
        config.Kd === undefined
      )
        return false;
    }

    if (needsKrumParams && !config.num_krum_selections) return false;
    if (needsTrimmedMeanParams && !config.trim_ratio) return false;

    if (needsGaussianParams) {
      if (
        config.gaussian_noise_mean === undefined ||
        config.gaussian_noise_std === undefined ||
        config.attack_ratio === undefined
      )
        return false;
    }

    return true;
  };

  const isValid = validateConfig();

  // Validation status per section
  const getSectionValidation = () => {
    const sections = {
      common: { valid: true, issues: [] },
      attack: { valid: true, issues: [] },
      strategy: { valid: true, issues: [] },
      resources: { valid: true, issues: [] },
      flower: { valid: true, issues: [] },
    };

    // Common Settings validation
    if (!config.aggregation_strategy_keyword) {
      sections.common.valid = false;
      sections.common.issues.push('Strategy required');
    }
    if (!config.dataset_keyword) {
      sections.common.valid = false;
      sections.common.issues.push('Dataset required');
    }
    if (config.num_of_rounds <= 0) {
      sections.common.valid = false;
      sections.common.issues.push('Rounds must be > 0');
    }
    if (config.num_of_clients <= 0) {
      sections.common.valid = false;
      sections.common.issues.push('Clients must be > 0');
    }
    if (!config.batch_size || config.batch_size <= 0) {
      sections.common.valid = false;
      sections.common.issues.push('Batch size required');
    }
    if (!config.num_of_client_epochs || config.num_of_client_epochs <= 0) {
      sections.common.valid = false;
      sections.common.issues.push('Client epochs required');
    }

    // Attack Configuration validation
    if (
      config.num_of_malicious_clients < 0 ||
      config.num_of_malicious_clients > config.num_of_clients
    ) {
      sections.attack.valid = false;
      sections.attack.issues.push('Invalid malicious client count');
    }
    if (config.num_of_malicious_clients > 0 && needsGaussianParams) {
      if (config.gaussian_noise_mean === undefined) {
        sections.attack.valid = false;
        sections.attack.issues.push('Gaussian mean required');
      }
      if (config.gaussian_noise_std === undefined) {
        sections.attack.valid = false;
        sections.attack.issues.push('Gaussian std required');
      }
      if (config.attack_ratio === undefined) {
        sections.attack.valid = false;
        sections.attack.issues.push('Attack ratio required');
      }
    }

    // Strategy-Specific Parameters validation
    if (needsTrustParams) {
      if (!config.begin_removing_from_round) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Begin removing round required');
      }
      if (!config.trust_threshold) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Trust threshold required');
      }
      if (!config.beta_value) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Beta value required');
      }
      if (!config.num_of_clusters) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Number of clusters required');
      }
    }

    if (needsPidParams) {
      if (config.num_std_dev === undefined) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Std dev threshold required');
      }
      if (config.Kp === undefined) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Kp required');
      }
      if (config.Ki === undefined) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Ki required');
      }
      if (config.Kd === undefined) {
        sections.strategy.valid = false;
        sections.strategy.issues.push('Kd required');
      }
    }

    if (needsKrumParams && !config.num_krum_selections) {
      sections.strategy.valid = false;
      sections.strategy.issues.push('Krum selections required');
    }

    if (needsTrimmedMeanParams && !config.trim_ratio) {
      sections.strategy.valid = false;
      sections.strategy.issues.push('Trim ratio required');
    }

    return sections;
  };

  const sectionValidation = getSectionValidation();

  // Smart section defaults based on context
  const getDefaultActiveKeys = useCallback(() => {
    const activeKeys = ['0']; // Common Settings always expanded

    // Attack Configuration: expand if attack preset or has malicious clients
    if (selectedPreset === 'attack' || config.num_of_malicious_clients > 0) {
      activeKeys.push('1');
    }

    // Strategy-Specific Parameters: expand if non-default strategy
    if (config.aggregation_strategy_keyword !== 'fedavg') {
      activeKeys.push('2');
    }

    // Resource & Output Settings: collapsed by default (eventKey="3")
    // Flower Framework Settings: collapsed by default (eventKey="4")

    return activeKeys;
  }, [selectedPreset, config.num_of_malicious_clients, config.aggregation_strategy_keyword]);

  const [activeKeys, setActiveKeys] = useState(getDefaultActiveKeys());

  // Update active keys when preset or config changes
  useEffect(() => {
    setActiveKeys(getDefaultActiveKeys());
  }, [getDefaultActiveKeys]);

  // Handle Esc key to navigate back to dashboard
  useEffect(() => {
    const handleEscape = e => {
      if (e.key === 'Escape') {
        navigate('/');
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [navigate]);

  // Generate plain-language summary
  const generateSummary = () => {
    const parts = [];

    if (config.num_of_malicious_clients > 0) {
      const attackName =
        config.attack_type === 'gaussian_noise' ? 'Gaussian noise' : 'label flipping';
      parts.push(`tests ${attackName} attacks with ${config.aggregation_strategy_keyword} defense`);
    } else {
      parts.push(
        `runs a benign federated learning simulation using ${config.aggregation_strategy_keyword} aggregation`
      );
    }

    parts.push(
      `over ${config.num_of_rounds} round${config.num_of_rounds !== 1 ? 's' : ''} using ${config.num_of_clients} client${config.num_of_clients !== 1 ? 's' : ''}`
    );

    if (config.num_of_malicious_clients > 0) {
      parts.push(`(${config.num_of_malicious_clients} malicious)`);
    }

    const datasetName =
      config.dataset_source === 'huggingface' ? config.hf_dataset_name : config.dataset_keyword;
    parts.push(`on ${datasetName} dataset`);

    if (config.remove_clients === 'true' && config.begin_removing_from_round) {
      parts.push(
        `Defense will begin removing suspicious clients from round ${config.begin_removing_from_round}`
      );
    }

    return 'This simulation ' + parts.join(' ') + '.';
  };

  // Estimate resource usage
  const estimateResources = () => {
    const totalOperations =
      config.num_of_rounds * config.num_of_clients * config.num_of_client_epochs;
    const estimatedMinutes = Math.ceil(totalOperations / 10); // rough estimate: 10 ops/min on CPU
    const estimatedMemoryMB = config.num_of_clients * 50; // rough estimate: 50MB per client
    const estimatedDiskMB = config.save_plots === 'true' || config.save_csv === 'true' ? 10 : 2;

    return { estimatedMinutes, estimatedMemoryMB, estimatedDiskMB };
  };

  // Check if value differs from default
  const isNonDefault = (key, value) => {
    return initialConfig[key] !== undefined && initialConfig[key] !== value;
  };

  // Export config as JSON
  const handleExportConfig = () => {
    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `simulation-config-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Import config from JSON
  const handleImportConfig = e => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      try {
        const imported = JSON.parse(event.target?.result);
        setConfig(imported);
        setSelectedPreset(null);
        alert('Config imported successfully!');
      } catch (err) {
        alert('Failed to parse JSON file. Please check the file format.');
        console.error('Import error:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <div>
      <div className="mb-3">
        <Button variant="outline-secondary" size="sm" as={Link} to="/">
          &larr; Back to Dashboard
        </Button>
        <span className="text-muted ms-2" style={{ fontSize: '0.85rem' }}>
          (or press Esc to cancel)
        </span>
      </div>
      <div className="d-flex align-items-center justify-content-between mb-2">
        <h1 className="mb-0">Create New Simulation</h1>
        <div className="d-flex align-items-center gap-2">
          {draftSaved && <span className="badge bg-success">Draft saved</span>}
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip>Export current configuration for reuse or sharing</Tooltip>}
          >
            <Button variant="outline-secondary" size="sm" onClick={handleExportConfig}>
              Export JSON
            </Button>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip>Import previously saved configuration</Tooltip>}
          >
            <Button
              variant="outline-secondary"
              size="sm"
              as="label"
              htmlFor="import-config"
              style={{ cursor: 'pointer', marginBottom: 0 }}
            >
              Import JSON
              <input
                type="file"
                id="import-config"
                accept=".json"
                style={{ display: 'none' }}
                onChange={handleImportConfig}
              />
            </Button>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip>Clear all changes and restore default values</Tooltip>}
          >
            <Button variant="outline-secondary" size="sm" onClick={handleClearDraft}>
              Reset to Defaults
            </Button>
          </OverlayTrigger>
        </div>
      </div>

      <div className="mb-4">
        <h5 className="mb-3">Choose a Preset Template</h5>
        <Row xs={1} md={2} lg={3} xl={5} className="g-3">
          {Object.entries(PRESETS).map(([key, preset]) => (
            <Col key={key}>
              <Card
                className={`h-100 ${selectedPreset === key ? 'border-primary' : ''}`}
                style={{
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  backgroundColor: selectedPreset === key ? 'rgba(13, 110, 253, 0.1)' : '',
                }}
                onClick={() => handlePresetChange(key)}
              >
                <Card.Body>
                  <div className="d-flex align-items-start gap-2 mb-2">
                    <span style={{ fontSize: '2rem' }}>{preset.icon}</span>
                    <div className="flex-grow-1">
                      <h6 className="mb-0">{preset.name}</h6>
                      <small className="text-muted">{preset.subtitle}</small>
                    </div>
                    {selectedPreset === key && (
                      <span style={{ fontSize: '1.5rem', color: '#0d6efd' }}>✓</span>
                    )}
                  </div>
                  <p className="small mb-2">{preset.description}</p>
                  <div className="text-muted small">
                    <strong>Est. time:</strong> {preset.estimatedTime}
                  </div>
                </Card.Body>
              </Card>
            </Col>
          ))}
          <Col>
            <Card
              className={`h-100 ${selectedPreset === null ? 'border-primary' : ''}`}
              style={{
                cursor: 'pointer',
                transition: 'all 0.2s',
                backgroundColor: selectedPreset === null ? 'rgba(13, 110, 253, 0.1)' : '',
              }}
              onClick={() => handlePresetChange(null)}
            >
              <Card.Body>
                <div className="d-flex align-items-start gap-2 mb-2">
                  <span style={{ fontSize: '2rem' }}>⚙️</span>
                  <div className="flex-grow-1">
                    <h6 className="mb-0">Custom</h6>
                    <small className="text-muted">Advanced configuration</small>
                  </div>
                  {selectedPreset === null && (
                    <span style={{ fontSize: '1.5rem', color: '#0d6efd' }}>✓</span>
                  )}
                </div>
                <p className="small mb-2">
                  Configure all parameters manually for specialized experiments
                </p>
                <div className="text-muted small">
                  <strong>Est. time:</strong> Varies
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </div>

      <Form onSubmit={handleSubmit}>
        <Accordion activeKey={activeKeys} onSelect={keys => setActiveKeys(keys)} alwaysOpen>
          <Accordion.Item eventKey="0">
            <Accordion.Header>
              <span
                className={sectionValidation.common.valid ? 'text-success' : 'text-danger'}
                style={{ marginRight: '8px' }}
              >
                {sectionValidation.common.valid ? '✓' : '✗'}
              </span>
              Common Settings{' '}
              <span className="text-muted ms-2" style={{ fontSize: '0.9em' }}>
                (6 required, 5 optional)
              </span>
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>
                  Strategy <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Aggregation algorithm for combining client model updates on the server.
                        "fedavg" (FedAvg) = simple averaging (no attack defense).
                        PID/Trust/Krum/etc. = Byzantine-robust strategies that detect and filter
                        malicious updates.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select
                  name="aggregation_strategy_keyword"
                  value={config.aggregation_strategy_keyword}
                  onChange={handleChange}
                >
                  {STRATEGIES.map(s => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Dataset Source <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Choose dataset source. Local = pre-configured datasets in this framework.
                        HuggingFace = download from HuggingFace Hub (100+ research datasets like
                        MNIST, CIFAR-10, FEMNIST with flexible partitioning).
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select
                  name="dataset_source"
                  value={config.dataset_source}
                  onChange={handleChange}
                >
                  <option value="local">Local Dataset</option>
                  <option value="huggingface">HuggingFace Hub</option>
                </Form.Select>
                {config.dataset_source === 'huggingface' && (
                  <Form.Text className="text-muted d-block mt-1">
                    💡 Browse more available datasets at{' '}
                    <a
                      href="https://huggingface.co/datasets"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ textDecoration: 'underline' }}
                    >
                      huggingface.co/datasets
                    </a>
                  </Form.Text>
                )}
              </Form.Group>

              {config.dataset_source === 'local' && (
                <Form.Group className="mb-3">
                  <Form.Label>
                    Dataset <span className="text-danger">*</span>{' '}
                    <OverlayTrigger
                      placement="right"
                      overlay={
                        <Tooltip>
                          Training dataset automatically split across clients. FEMNIST = handwritten
                          characters (image classification). Medical: pneumoniamnist, bloodmnist,
                          lung_photos = medical imaging. FLAIR, MedQuAD = medical text (NLP). IID =
                          evenly distributed, NIID = non-evenly distributed (more realistic).
                        </Tooltip>
                      }
                    >
                      <span style={{ cursor: 'help' }}>ℹ️</span>
                    </OverlayTrigger>
                  </Form.Label>
                  <Form.Select
                    name="dataset_keyword"
                    value={config.dataset_keyword}
                    onChange={handleChange}
                  >
                    {DATASETS.map(d => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
              )}

              {config.dataset_source === 'huggingface' && (
                <>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      HuggingFace Dataset Name <span className="text-danger">*</span>{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Dataset identifier from HuggingFace Hub. Select from suggestions or type
                            custom dataset. Examples: "ylecun/mnist", "uoft-cs/cifar10". Real-time
                            validation checks if dataset exists.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="text"
                      name="hf_dataset_name"
                      value={config.hf_dataset_name}
                      onChange={handleChange}
                      list="popular-datasets"
                      placeholder="Select from suggestions or type dataset name..."
                    />
                    <datalist id="popular-datasets">
                      {POPULAR_DATASETS.map(d => (
                        <option key={d.value} value={d.value}>
                          {d.label}
                        </option>
                      ))}
                    </datalist>
                    {datasetValidation.loading && (
                      <Form.Text className="text-muted d-block mt-1">
                        ⏳ Checking dataset...
                      </Form.Text>
                    )}
                    {datasetValidation.valid === false && (
                      <Form.Text className="text-danger d-block mt-1">
                        ❌ Dataset not found: {datasetValidation.error}
                      </Form.Text>
                    )}
                    {datasetValidation.valid === true && !datasetValidation.compatible && (
                      <Form.Text className="text-warning d-block mt-1">
                        ⚠️ Dataset found but may not be compatible with Flower Datasets
                      </Form.Text>
                    )}
                    {datasetValidation.valid === true && datasetValidation.compatible && (
                      <Form.Text className="text-success d-block mt-1">
                        ✅ Valid dataset ({datasetValidation.info?.num_examples?.toLocaleString()}{' '}
                        examples, splits: {datasetValidation.info?.splits?.join(', ')})
                        {datasetValidation.info?.key_features?.length > 0 && (
                          <span> • Fields: {datasetValidation.info.key_features.join(', ')}</span>
                        )}
                      </Form.Text>
                    )}
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      Partitioning Strategy <span className="text-danger">*</span>{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            How to distribute data across clients. IID = balanced/uniform. Dirichlet
                            = realistic heterogeneous distribution (tune α: lower = more
                            heterogeneous). Pathological = extreme non-IID (each client gets limited
                            label classes).
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Select
                      name="partitioning_strategy"
                      value={config.partitioning_strategy}
                      onChange={handleChange}
                    >
                      <option value="iid">IID (Balanced)</option>
                      <option value="dirichlet">Dirichlet (Heterogeneous)</option>
                      <option value="pathological">Pathological (Extreme Non-IID)</option>
                    </Form.Select>
                    {datasetValidation.valid &&
                      datasetValidation.info?.has_label === false &&
                      (config.partitioning_strategy === 'dirichlet' ||
                        config.partitioning_strategy === 'pathological') && (
                        <Form.Text
                          className="d-block mt-1"
                          style={{ color: '#d97706', fontWeight: '500' }}
                        >
                          ⚠️ Warning: This dataset may not work with {config.partitioning_strategy}{' '}
                          partitioning (no "label" field detected). Consider using IID partitioning
                          or switching to a classification dataset.
                        </Form.Text>
                      )}
                    {datasetValidation.valid &&
                      datasetValidation.info?.has_label === false &&
                      config.partitioning_strategy === 'iid' && (
                        <Form.Text
                          className="d-block mt-1"
                          style={{ color: '#0ea5e9', fontWeight: '500' }}
                        >
                          ℹ️ This dataset works best with IID partitioning since no label field was
                          detected. Good choice!
                        </Form.Text>
                      )}
                  </Form.Group>

                  {config.partitioning_strategy === 'dirichlet' && (
                    <Form.Group className="mb-3">
                      <Form.Label>
                        Dirichlet Alpha (α){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Controls data heterogeneity. Lower α = more heterogeneous (realistic).
                              α=0.1 = very heterogeneous, α=0.5 = moderate, α=10.0 = nearly IID.
                              Typical research values: 0.1-1.0.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>{' '}
                        <span className="text-muted">
                          Current: {config.partitioning_params?.alpha || 0.5}
                        </span>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        min="0.01"
                        max="100"
                        value={config.partitioning_params?.alpha || 0.5}
                        onChange={e => {
                          const newParams = {
                            ...config.partitioning_params,
                            alpha: parseFloat(e.target.value),
                          };
                          handleChange({
                            target: { name: 'partitioning_params', value: newParams },
                          });
                        }}
                      />
                    </Form.Group>
                  )}
                </>
              )}

              <Form.Group className="mb-3">
                <Form.Label>
                  Model Type{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Neural network architecture. CNN (Convolutional Neural Network) = best for
                        images (fast, efficient). Transformer = best for text/NLP tasks (required
                        for LLM finetuning, slower but more accurate for language).
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select name="model_type" value={config.model_type} onChange={handleChange}>
                  <option value="cnn">CNN</option>
                  <option value="transformer">Transformer</option>
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Use LLM{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Enable Large Language Model (LLM) finetuning for medical NLP tasks. Only
                            works with Transformer model type. Uses BiomedBERT for medical text
                            understanding. Requires medical text dataset (FLAIR, MedQuAD).
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="use_llm"
                  checked={config.use_llm === 'true'}
                  onChange={e =>
                    setConfig(prev => ({ ...prev, use_llm: e.target.checked ? 'true' : 'false' }))
                  }
                />
                {getFieldError('use_llm') && (
                  <Form.Text className="text-danger d-block">
                    ❌ {getFieldError('use_llm').message}
                  </Form.Text>
                )}
              </Form.Group>

              {needsLlmParams && (
                <>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      LLM Model{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Pre-trained language model for medical NLP. BiomedBERT = specialized for
                            biomedical text understanding, trained on PubMed abstracts and medical
                            literature. Better than general models (BERT, GPT) for medical
                            terminology and concepts.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Select name="llm_model" value={config.llm_model} onChange={handleChange}>
                      <option value="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext">
                        BiomedBERT
                      </option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      LLM Fine-tuning{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Fine-tuning method. "Full" = update all model parameters (slow,
                            resource-intensive, best accuracy). "LoRA" (Low-Rank Adaptation) = only
                            update small adapter layers (fast, efficient, 90% of full performance
                            with 1% of parameters). Recommended: LoRA for federated learning.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Select
                      name="llm_finetuning"
                      value={config.llm_finetuning}
                      onChange={handleChange}
                    >
                      <option value="full">Full</option>
                      <option value="lora">LoRA</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      LLM Task{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Training objective. MLM (Masked Language Modeling) = randomly mask words
                            and predict them from context. Teaches model to understand medical
                            language structure. Example: "The patient has [MASK] diabetes" → predict
                            "Type 2". Standard pretraining task for BERT-style models.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Select name="llm_task" value={config.llm_task} onChange={handleChange}>
                      <option value="mlm">MLM (Masked Language Modeling)</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      LLM Chunk Size{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Maximum sequence length (number of tokens) for text input. Longer text
                            is split into chunks. Higher = more context per chunk but slower
                            processing and more memory. BiomedBERT max: 512 tokens. Typical:
                            128-512. Quick test: 256.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      name="llm_chunk_size"
                      value={config.llm_chunk_size}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('llm_chunk_size')}
                    />
                    {getFieldError('llm_chunk_size') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('llm_chunk_size').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>

                  {needsMLMParams && (
                    <Form.Group className="mb-3">
                      <Form.Label>
                        MLM Probability{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Fraction of input tokens to mask during MLM training (0-1). 0.15 =
                              mask 15% of words (BERT standard). Higher = more challenging task (may
                              improve learning) but harder to converge. Lower = easier but less
                              effective. Typical: 0.10-0.20. Standard: 0.15.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="mlm_probability"
                        value={config.mlm_probability}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('mlm_probability')}
                      />
                      {getFieldError('mlm_probability') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('mlm_probability').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>
                  )}
                </>
              )}

              <Form.Group className="mb-3">
                <Form.Label>
                  Number of Rounds <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Communication rounds between server and clients. Each round: clients train
                        locally → send updates to server → server aggregates → broadcasts new global
                        model. More rounds = better convergence but longer runtime. Quick test: 2-5
                        rounds. Research: 10-20 rounds.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="num_of_rounds"
                  value={config.num_of_rounds}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Number of Clients <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Total participating devices/clients in the simulation. Each client gets a
                        portion of the dataset. More clients = more realistic federated learning
                        (simulates hospitals, phones, etc.) but slower simulation. Quick test: 3-5
                        clients. Research: 10-100 clients.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="num_of_clients"
                  value={config.num_of_clients}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Batch Size <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Number of training examples processed together in each gradient update.
                        Larger batches = faster training but need more memory and may reduce model
                        accuracy. Smaller batches = slower but more fine-grained learning. Standard:
                        16-64. Quick test: 20-32.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="batch_size"
                  value={config.batch_size}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Client Epochs <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Number of complete passes through the local dataset each client makes before
                        sending updates to the server. 1 epoch = see each training example once.
                        More epochs = better local learning but longer per-round time and risk of
                        overfitting to local data. Quick test: 1-2. Research: 1-5.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="num_of_client_epochs"
                  value={config.num_of_client_epochs}
                  onChange={handleChange}
                />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="1">
            <Accordion.Header>
              <span
                className={sectionValidation.attack.valid ? 'text-success' : 'text-danger'}
                style={{ marginRight: '8px' }}
              >
                {sectionValidation.attack.valid ? '✓' : '✗'}
              </span>
              Attack Configuration
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>
                  Number of Malicious Clients{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Number of Byzantine (malicious/compromised) clients that send corrupted
                        model updates. These simulate real-world attacks like poisoning or sabotage.
                        0 = benign simulation. Set to 1-3 to test defense strategies. Must be &lt;
                        total clients.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="num_of_malicious_clients"
                  value={config.num_of_malicious_clients}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Attack Type{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Type of Byzantine attack malicious clients perform. "gaussian_noise" = add
                        random noise to model weights (simulates faulty sensors or corruption).
                        "label_flipping" = flip training labels to poison the model. Only applies if
                        malicious clients &gt; 0.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select name="attack_type" value={config.attack_type} onChange={handleChange}>
                  {ATTACKS.map(a => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>

              {needsGaussianParams && (
                <>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Gaussian Noise Mean{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Center point of the Gaussian (normal) distribution used for noise
                            attack. 0 = noise centered at zero (adds both positive and negative
                            corruption). Higher values = more positive bias in corruption. Typical:
                            0.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      name="gaussian_noise_mean"
                      value={config.gaussian_noise_mean || 0}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('gaussian_noise_mean')}
                    />
                    {getFieldError('gaussian_noise_mean') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('gaussian_noise_mean').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      Gaussian Noise Std{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Standard deviation (spread) of Gaussian noise added to model weights.
                            Higher values = more severe attack, larger corruption. 1-10 = mild.
                            50-100 = severe (makes model unusable). Test defense strength by varying
                            this.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      name="gaussian_noise_std"
                      value={config.gaussian_noise_std || 1}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('gaussian_noise_std')}
                    />
                    {getFieldError('gaussian_noise_std') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('gaussian_noise_std').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      Attack Ratio{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Fraction of malicious client's model weights to corrupt (0-1). 1.0 =
                            attack all weights (full corruption). 0.5 = attack half. Lower ratios =
                            subtler, harder-to-detect attacks. Typical: 0.5-1.0.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      step="0.01"
                      name="attack_ratio"
                      value={config.attack_ratio || 0.5}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('attack_ratio')}
                    />
                    {getFieldError('attack_ratio') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('attack_ratio').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>
                </>
              )}
            </Accordion.Body>
          </Accordion.Item>

          {(needsTrustParams || needsPidParams || needsKrumParams || needsTrimmedMeanParams) && (
            <Accordion.Item eventKey="2">
              <Accordion.Header>
                <span
                  className={sectionValidation.strategy.valid ? 'text-success' : 'text-danger'}
                  style={{ marginRight: '8px' }}
                >
                  {sectionValidation.strategy.valid ? '✓' : '✗'}
                </span>
                Strategy-Specific Parameters
              </Accordion.Header>
              <Accordion.Body>
                {needsTrustParams && (
                  <>
                    <Form.Group className="mb-3">
                      <Form.Label>
                        Begin Removing From Round{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Round number when trust strategy starts removing low-trust clients.
                              Early rounds build trust scores. Later rounds filter out malicious
                              clients. Earlier start = more aggressive (may remove good clients).
                              Later start = let trust stabilize first. Typical: 2-5.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        name="begin_removing_from_round"
                        value={config.begin_removing_from_round || 1}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('begin_removing_from_round')}
                      />
                      {getFieldError('begin_removing_from_round') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('begin_removing_from_round').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Trust Threshold{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Minimum trust score (0-1) required for client to participate after
                              removal begins. Clients below this are excluded as potentially
                              malicious. Lower (0.1-0.3) = permissive, only remove obvious
                              attackers. Higher (0.5-0.7) = strict, remove any suspicious clients.
                              Typical: 0.15-0.5.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="trust_threshold"
                        value={config.trust_threshold || 0.5}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('trust_threshold')}
                      />
                      {getFieldError('trust_threshold') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('trust_threshold').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Beta Value{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Exponential moving average weight for trust score updates (0-1).
                              Higher (0.9-0.99) = trust changes slowly, prioritize history. Lower
                              (0.5-0.8) = trust adapts quickly to recent behavior. Controls
                              responsiveness vs. stability of trust scores. Typical: 0.75-0.9.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="beta_value"
                        value={config.beta_value || 0.9}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('beta_value')}
                      />
                      {getFieldError('beta_value') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('beta_value').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Number of Clusters{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Number of client groups for clustering-based trust analysis. Clients
                              are grouped by update similarity. More clusters = finer-grained
                              analysis but higher complexity. 1 = simple (all clients in one group).
                              Note: Currently limited to 1 in this implementation.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        name="num_of_clusters"
                        value={config.num_of_clusters || 1}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('num_of_clusters')}
                      />
                      {getFieldError('num_of_clusters') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('num_of_clusters').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>
                  </>
                )}

                {needsPidParams && (
                  <>
                    <Form.Group className="mb-3">
                      <Form.Label>
                        Number of Std Deviations{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              Statistical threshold for outlier detection. Client updates beyond
                              this many standard deviations from the mean are considered outliers
                              and adaptively filtered. Higher = more permissive (fewer false
                              positives). Lower = stricter (catches subtler attacks). Typical:
                              1.5-3.0. Standard: 2.0 (95% of normal data).
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="num_std_dev"
                        value={config.num_std_dev || 2.0}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('num_std_dev')}
                      />
                      {getFieldError('num_std_dev') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('num_std_dev').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Kp (Proportional Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              PID controller proportional term - reacts to current outlier detection
                              error. Higher = more aggressive threshold adjustments (faster response
                              but may oscillate). Lower = gentler adjustments (more stable).
                              Controls how quickly the defense adapts. Typical: 0.5-2.0.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="Kp"
                        value={config.Kp || 1.0}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('Kp')}
                      />
                      {getFieldError('Kp') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('Kp').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Ki (Integral Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              PID controller integral term - eliminates long-term bias by
                              accumulating past errors. Ensures threshold converges to optimal level
                              over time. Higher = faster convergence but may cause instability.
                              Lower = slower, more stable. Typically small: 0.01-0.1.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="Ki"
                        value={config.Ki || 0.1}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('Ki')}
                      />
                      {getFieldError('Ki') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('Ki').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Kd (Derivative Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={
                            <Tooltip>
                              PID controller derivative term - predicts future errors based on rate
                              of change. Dampens oscillations and prevents overshooting. Higher =
                              more damping (smoother but slower response). Lower = less damping
                              (faster but may oscillate). Typically very small: 0.001-0.05.
                            </Tooltip>
                          }
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control
                        type="number"
                        step="0.01"
                        name="Kd"
                        value={config.Kd || 0.01}
                        onChange={handleChange}
                        isInvalid={!!getFieldError('Kd')}
                      />
                      {getFieldError('Kd') && (
                        <Form.Control.Feedback type="invalid">
                          {getFieldError('Kd').message}
                        </Form.Control.Feedback>
                      )}
                    </Form.Group>
                  </>
                )}

                {needsKrumParams && (
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Krum Selections{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Number of clients with most similar updates to select for aggregation
                            (based on Euclidean distance). Krum selects the "closest neighbors" and
                            ignores outliers. Lower = more Byzantine robustness (strict filtering)
                            but less data diversity. Higher = more data but less robust. Must be
                            &lt; total clients. Typical: (num_clients - malicious_clients - 2).
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      name="num_krum_selections"
                      value={config.num_krum_selections || 5}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('num_krum_selections')}
                    />
                    {getFieldError('num_krum_selections') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('num_krum_selections').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>
                )}

                {needsTrimmedMeanParams && (
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Trim Ratio{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Fraction of most extreme (highest and lowest) values to remove before
                            averaging (0-0.5). 0.1 = remove top 10% and bottom 10%, average the
                            middle 80%. Higher = more aggressive outlier filtering (robust against
                            attacks) but loses more data. Lower = keep more data but less robust.
                            Must be &lt; 0.5. Typical: 0.1-0.3.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      step="0.01"
                      name="trim_ratio"
                      value={config.trim_ratio || 0.1}
                      onChange={handleChange}
                      isInvalid={!!getFieldError('trim_ratio')}
                    />
                    {getFieldError('trim_ratio') && (
                      <Form.Control.Feedback type="invalid">
                        {getFieldError('trim_ratio').message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>
                )}
              </Accordion.Body>
            </Accordion.Item>
          )}

          <Accordion.Item eventKey="3">
            <Accordion.Header>
              <span
                className={sectionValidation.resources.valid ? 'text-success' : 'text-danger'}
                style={{ marginRight: '8px' }}
              >
                {sectionValidation.resources.valid ? '✓' : '✗'}
              </span>
              Resource & Output Settings
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>
                  Training Device{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Hardware for model training. "cpu" = uses CPU only (slower but works
                        everywhere). "gpu" or "cuda" = uses GPU acceleration (10-100x faster but
                        requires NVIDIA GPU). For quick tests, CPU is fine. For large
                        models/datasets, GPU highly recommended.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select
                  name="training_device"
                  value={config.training_device}
                  onChange={handleChange}
                >
                  {DEVICES.map(d => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  CPUs per Client{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Number of CPU cores allocated to each client for parallel processing. 1 =
                        single-threaded (slowest). Higher = faster training but uses more system
                        resources. Limited by your machine's CPU count. Typical: 1-4.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="cpus_per_client"
                  value={config.cpus_per_client}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  GPUs per Client{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Fraction of GPU memory allocated to each client (0-1). 0 = CPU only. 1.0 =
                        full GPU. 0.1-0.3 = shared GPU (multiple clients). Useful for simulating
                        multiple devices on one GPU. Requires "gpu" or "cuda" training device.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  name="gpus_per_client"
                  value={config.gpus_per_client}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Training Subset Fraction{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Fraction of dataset to use for training (0-1). 1.0 = use full dataset (most
                        accurate). 0.1-0.5 = use subset (faster experiments). Reduces data per
                        client proportionally. Useful for quick testing without waiting for full
                        training. Typical: 0.5-1.0. Quick test: 0.1-0.3.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  name="training_subset_fraction"
                  value={config.training_subset_fraction}
                  onChange={handleChange}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Show Plots{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Display matplotlib plots in popup windows during simulation (accuracy,
                            loss curves, etc.). Useful for real-time monitoring but may slow down
                            headless servers. Enable for interactive use, disable for background
                            jobs.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="show_plots"
                  checked={config.show_plots === 'true'}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      show_plots: e.target.checked ? 'true' : 'false',
                    }))
                  }
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Save Plots{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Save plots as PNG/PDF files in output directory. Includes accuracy
                            curves, loss graphs, and strategy-specific visualizations (trust scores,
                            PID thresholds). Always recommended for record-keeping and paper
                            figures. Minimal overhead.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="save_plots"
                  checked={config.save_plots === 'true'}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      save_plots: e.target.checked ? 'true' : 'false',
                    }))
                  }
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Save CSV{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Export results as CSV files (metrics, per-round statistics, per-client
                            data). Useful for custom analysis, Excel import, or statistical
                            processing. Enable for research experiments. Small file size (~KB).
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="save_csv"
                  checked={config.save_csv === 'true'}
                  onChange={e =>
                    setConfig(prev => ({ ...prev, save_csv: e.target.checked ? 'true' : 'false' }))
                  }
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Preserve Dataset{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Keep downloaded/preprocessed datasets cached on disk for future runs.
                            Enabled = faster subsequent simulations (no re-download). Disabled =
                            re-download each time (uses more bandwidth). Recommended: keep enabled
                            unless testing dataset loading.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="preserve_dataset"
                  checked={config.preserve_dataset === 'true'}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      preserve_dataset: e.target.checked ? 'true' : 'false',
                    }))
                  }
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Remove Clients{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Enable client removal for Byzantine-robust strategies (Trust, PID). When
                            enabled, strategies can permanently exclude low-trust/malicious clients
                            after threshold rounds. Disabled = all clients always participate
                            (useful for benign baselines). Recommended: enable when testing
                            defenses.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="remove_clients"
                  checked={config.remove_clients === 'true'}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      remove_clients: e.target.checked ? 'true' : 'false',
                    }))
                  }
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label={
                    <>
                      Strict Mode{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={
                          <Tooltip>
                            Enforce all clients participate every round (auto-sets
                            min_fit/evaluate/available_clients = num_clients). Enabled = full
                            synchronous FL (realistic, prevents stragglers). Disabled = allow
                            partial participation (flexible but less common). Recommended: keep
                            enabled for standard FL experiments.
                          </Tooltip>
                        }
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </>
                  }
                  name="strict_mode"
                  checked={config.strict_mode === 'true'}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      strict_mode: e.target.checked ? 'true' : 'false',
                    }))
                  }
                />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="4">
            <Accordion.Header>
              <span
                className={sectionValidation.flower.valid ? 'text-success' : 'text-danger'}
                style={{ marginRight: '8px' }}
              >
                {sectionValidation.flower.valid ? '✓' : '✗'}
              </span>
              Flower Framework Settings
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>
                  Min Fit Clients{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Minimum clients that must participate in training ("fit" = train model) each
                        round. Lower values allow rounds to proceed with fewer clients, but may
                        reduce model quality. Must be ≤ total clients.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="min_fit_clients"
                  value={config.min_fit_clients}
                  onChange={handleChange}
                  isInvalid={!!getFieldError('min_fit_clients')}
                  isValid={
                    !getFieldError('min_fit_clients') &&
                    !getFieldWarning('min_fit_clients') &&
                    config.min_fit_clients !== ''
                  }
                />
                {getFieldError('min_fit_clients') && (
                  <Form.Control.Feedback type="invalid">
                    {getFieldError('min_fit_clients').message}
                  </Form.Control.Feedback>
                )}
                {getFieldWarning('min_fit_clients') && !getFieldError('min_fit_clients') && (
                  <Form.Text className="text-warning">
                    ⚠️ {getFieldWarning('min_fit_clients').message}
                  </Form.Text>
                )}
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Min Evaluate Clients{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Minimum clients that must participate in evaluation (testing model accuracy)
                        each round. Evaluation happens after training to measure global model
                        performance. Must be ≤ total clients.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="min_evaluate_clients"
                  value={config.min_evaluate_clients}
                  onChange={handleChange}
                  isInvalid={!!getFieldError('min_evaluate_clients')}
                  isValid={
                    !getFieldError('min_evaluate_clients') &&
                    !getFieldWarning('min_evaluate_clients') &&
                    config.min_evaluate_clients !== ''
                  }
                />
                {getFieldError('min_evaluate_clients') && (
                  <Form.Control.Feedback type="invalid">
                    {getFieldError('min_evaluate_clients').message}
                  </Form.Control.Feedback>
                )}
                {getFieldWarning('min_evaluate_clients') &&
                  !getFieldError('min_evaluate_clients') && (
                    <Form.Text className="text-warning">
                      ⚠️ {getFieldWarning('min_evaluate_clients').message}
                    </Form.Text>
                  )}
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Min Available Clients{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={
                      <Tooltip>
                        Minimum clients that must be connected and ready before a round can start.
                        Server waits until this many clients are available. Simulates real-world
                        scenarios where devices may be offline. Must be ≤ total clients.
                      </Tooltip>
                    }
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control
                  type="number"
                  name="min_available_clients"
                  value={config.min_available_clients}
                  onChange={handleChange}
                  isInvalid={!!getFieldError('min_available_clients')}
                  isValid={
                    !getFieldError('min_available_clients') &&
                    !getFieldWarning('min_available_clients') &&
                    config.min_available_clients !== ''
                  }
                />
                {getFieldError('min_available_clients') && (
                  <Form.Control.Feedback type="invalid">
                    {getFieldError('min_available_clients').message}
                  </Form.Control.Feedback>
                )}
                {getFieldWarning('min_available_clients') &&
                  !getFieldError('min_available_clients') && (
                    <Form.Text className="text-warning">
                      ⚠️ {getFieldWarning('min_available_clients').message}
                    </Form.Text>
                  )}
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Evaluate Metrics Aggregation Function</Form.Label>
                <Form.Control
                  type="text"
                  name="evaluate_metrics_aggregation_fn"
                  value={config.evaluate_metrics_aggregation_fn}
                  onChange={handleChange}
                />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>
        </Accordion>

        {/* Validation Summary */}
        <div className="mt-4">
          <ValidationSummary errors={errors} warnings={warnings} infos={infos} />
        </div>

        {isValid && (
          <Card className="mt-4 bg-light border-primary">
            <Card.Body>
              <h6 className="mb-3">📋 Configuration Summary</h6>
              <p className="mb-3 text-muted">{generateSummary()}</p>

              <Row className="g-3">
                <Col md={4}>
                  <div className="small">
                    <strong>⏱️ Estimated Time:</strong>{' '}
                    <span
                      className={
                        isNonDefault('num_of_rounds', config.num_of_rounds) ||
                        isNonDefault('num_of_clients', config.num_of_clients)
                          ? 'fw-bold text-primary'
                          : ''
                      }
                    >
                      ~{estimateResources().estimatedMinutes} min
                    </span>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="small">
                    <strong>💾 Memory Usage:</strong>{' '}
                    <span
                      className={
                        isNonDefault('num_of_clients', config.num_of_clients)
                          ? 'fw-bold text-primary'
                          : ''
                      }
                    >
                      ~{estimateResources().estimatedMemoryMB} MB
                    </span>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="small">
                    <strong>💿 Disk Space:</strong>{' '}
                    <span
                      className={
                        isNonDefault('save_plots', config.save_plots) ||
                        isNonDefault('save_csv', config.save_csv)
                          ? 'fw-bold text-primary'
                          : ''
                      }
                    >
                      ~{estimateResources().estimatedDiskMB} MB
                    </span>
                  </div>
                </Col>
              </Row>

              {(isNonDefault('num_of_rounds', config.num_of_rounds) ||
                isNonDefault('num_of_clients', config.num_of_clients) ||
                isNonDefault('aggregation_strategy_keyword', config.aggregation_strategy_keyword) ||
                isNonDefault('dataset_keyword', config.dataset_keyword)) && (
                <div className="mt-3 pt-3 border-top">
                  <div className="small text-muted">
                    <strong>Modified from defaults:</strong>{' '}
                    {[
                      isNonDefault(
                        'aggregation_strategy_keyword',
                        config.aggregation_strategy_keyword
                      ) && `Strategy (${config.aggregation_strategy_keyword})`,
                      isNonDefault('dataset_keyword', config.dataset_keyword) &&
                        `Dataset (${config.dataset_keyword})`,
                      isNonDefault('num_of_rounds', config.num_of_rounds) &&
                        `Rounds (${config.num_of_rounds})`,
                      isNonDefault('num_of_clients', config.num_of_clients) &&
                        `Clients (${config.num_of_clients})`,
                    ]
                      .filter(Boolean)
                      .join(', ')}
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>
        )}

        <div className="mt-3">
          <div className="d-flex align-items-center gap-3">
            <Button variant="primary" type="submit" disabled={submitting || !validation.isValid}>
              {submitting
                ? 'Launching...'
                : errors.length > 0
                  ? `Fix ${errors.length} error${errors.length > 1 ? 's' : ''} to launch`
                  : 'Launch Simulation'}
            </Button>
            {!validation.isValid && errors.length > 0 && (
              <span className="badge bg-danger">
                {errors.length} error{errors.length !== 1 ? 's' : ''} remaining
              </span>
            )}
            {validation.isValid && warnings.length > 0 && (
              <span className="badge bg-warning">
                {warnings.length} warning{warnings.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          {!validation.isValid && errors.length > 0 && (
            <div className="text-muted mt-2">
              Please fix all validation errors to launch simulation
            </div>
          )}
        </div>

        {error && (
          <Alert variant="danger" className="mt-3">
            {error}
          </Alert>
        )}
      </Form>
    </div>
  );
}

export default NewSimulation;
