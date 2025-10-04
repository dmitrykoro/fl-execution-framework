import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Form, Button, Card, Alert, Accordion, OverlayTrigger, Tooltip, Row, Col } from 'react-bootstrap';
import { createSimulation } from '../api';

// Defaults from config/simulation_strategies/example_strategy_config.json
const initialConfig = {
  // Core Simulation Settings
  num_of_rounds: 4,
  num_of_clients: 5,
  num_of_malicious_clients: 1,
  attack_ratio: 1.0,

  // Dataset & Model
  dataset_keyword: "femnist_iid",
  model_type: "cnn",
  use_llm: "false",

  // Attack Configuration
  attack_type: "gaussian_noise",
  gaussian_noise_mean: 0,
  gaussian_noise_std: 75,
  num_std_dev: 2,

  // Defense Strategy
  aggregation_strategy_keyword: "pid",
  remove_clients: "true",
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
  training_device: "cpu",
  cpus_per_client: 1,
  gpus_per_client: 0.0,
  num_of_client_epochs: 1,
  batch_size: 20,
  training_subset_fraction: 0.9,

  // Client Requirements
  min_fit_clients: 5,
  min_evaluate_clients: 5,
  min_available_clients: 5,
  evaluate_metrics_aggregation_fn: "weighted_average",

  // Output Settings
  show_plots: "false",
  save_plots: "true",
  save_csv: "true",
  preserve_dataset: "true",
  strict_mode: "true",

  // LLM Settings
  llm_model: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
  llm_finetuning: "lora",
  llm_task: "mlm",
  llm_chunk_size: 256,
  mlm_probability: 0.15,
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.1,
  lora_target_modules: ["query", "value"]
};

const STRATEGIES = ["fedavg", "trust", "pid", "pid_scaled", "pid_standardized", "multi-krum", "krum", "multi-krum-based", "trimmed_mean", "rfa", "bulyan"];
const DATASETS = ["femnist_iid", "femnist_niid", "its", "pneumoniamnist", "flair", "bloodmnist", "medquad", "lung_photos"];
const ATTACKS = ["gaussian_noise", "label_flipping"];
const DEVICES = ["cpu", "gpu", "cuda"];

const PRESETS = {
  quick: {
    name: "Quick Test",
    subtitle: "2 rounds / 3 clients",
    description: "Fast test for validating setup and basic functionality",
    estimatedTime: "15-20 seconds",
    icon: "⚡",
    config: {
      num_of_rounds: 2,
      num_of_clients: 3,
      num_of_malicious_clients: 0,
      min_fit_clients: 3,
      min_evaluate_clients: 3,
      min_available_clients: 3,
    }
  },
  full: {
    name: "Full Run",
    subtitle: "10 rounds / 10 clients",
    description: "Complete simulation for research experiments and demonstrations",
    estimatedTime: "2-3 minutes",
    icon: "🔬",
    config: {
      num_of_rounds: 10,
      num_of_clients: 10,
      num_of_malicious_clients: 0,
      min_fit_clients: 10,
      min_evaluate_clients: 10,
      min_available_clients: 10,
    }
  },
  attack: {
    name: "Attack Test",
    subtitle: "5 rounds / 2 malicious",
    description: "Test Byzantine attack detection with malicious clients",
    estimatedTime: "45-60 seconds",
    icon: "⚔️",
    config: {
      num_of_rounds: 5,
      num_of_clients: 5,
      num_of_malicious_clients: 2,
      min_fit_clients: 5,
      min_evaluate_clients: 5,
      min_available_clients: 5,
    }
  },
  convergence: {
    name: "Convergence Test",
    subtitle: "15 rounds / 8 clients",
    description: "Optimized for smooth plot curves and visualization testing with PID defense tracking",
    estimatedTime: "60-90 seconds",
    icon: "📈",
    config: {
      num_of_rounds: 15,
      num_of_clients: 8,
      num_of_malicious_clients: 0,
      aggregation_strategy_keyword: "pid",
      min_fit_clients: 8,
      min_evaluate_clients: 8,
      min_available_clients: 8,
    }
  }
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

  // Auto-save to localStorage when config changes
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      localStorage.setItem('simulation-draft', JSON.stringify(config));
      setDraftSaved(true);
      setTimeout(() => setDraftSaved(false), 2000);
    }, 1000); // Debounce saves by 1 second

    return () => clearTimeout(timeoutId);
  }, [config]);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let finalValue = value;

    if (type === 'number') {
      finalValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    }

    setConfig(prev => ({ ...prev, [name]: finalValue }));
  };

  const handlePresetChange = (presetKey) => {
    if (presetKey && PRESETS[presetKey]) {
      setSelectedPreset(presetKey);
      setConfig(prev => ({ ...prev, ...PRESETS[presetKey].config }));
    } else {
      setSelectedPreset(null);
    }
  };

  const handleSubmit = async (e) => {
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
      console.error("Failed to create simulation:", err);
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

  const needsTrustParams = config.aggregation_strategy_keyword === "trust";
  const needsPidParams = ["pid", "pid_scaled", "pid_standardized"].includes(config.aggregation_strategy_keyword);
  const needsKrumParams = ["multi-krum", "krum", "multi-krum-based"].includes(config.aggregation_strategy_keyword);
  const needsTrimmedMeanParams = config.aggregation_strategy_keyword === "trimmed_mean";
  const needsGaussianParams = config.attack_type === "gaussian_noise";
  const needsLlmParams = config.use_llm === "true";
  const needsMLMParams = needsLlmParams && config.llm_task === "mlm";

  const validateConfig = () => {
    if (!config.aggregation_strategy_keyword || !config.dataset_keyword) return false;
    if (config.num_of_rounds <= 0 || config.num_of_clients <= 0) return false;
    if (config.num_of_malicious_clients < 0 || config.num_of_malicious_clients > config.num_of_clients) return false;

    if (needsTrustParams) {
      if (!config.begin_removing_from_round || !config.trust_threshold || !config.beta_value || !config.num_of_clusters) return false;
    }

    if (needsPidParams) {
      if (config.num_std_dev === undefined || config.Kp === undefined || config.Ki === undefined || config.Kd === undefined) return false;
    }

    if (needsKrumParams && !config.num_krum_selections) return false;
    if (needsTrimmedMeanParams && !config.trim_ratio) return false;

    if (needsGaussianParams) {
      if (config.gaussian_noise_mean === undefined || config.gaussian_noise_std === undefined || config.attack_ratio === undefined) return false;
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
      flower: { valid: true, issues: [] }
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
    if (config.num_of_malicious_clients < 0 || config.num_of_malicious_clients > config.num_of_clients) {
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
  const totalIssues = Object.values(sectionValidation).reduce((sum, section) => sum + section.issues.length, 0);

  // Smart section defaults based on context
  const getDefaultActiveKeys = () => {
    const activeKeys = ["0"]; // Common Settings always expanded

    // Attack Configuration: expand if attack preset or has malicious clients
    if (selectedPreset === "attack" || config.num_of_malicious_clients > 0) {
      activeKeys.push("1");
    }

    // Strategy-Specific Parameters: expand if non-default strategy
    if (config.aggregation_strategy_keyword !== "fedavg") {
      activeKeys.push("2");
    }

    // Resource & Output Settings: collapsed by default (eventKey="3")
    // Flower Framework Settings: collapsed by default (eventKey="4")

    return activeKeys;
  };

  const [activeKeys, setActiveKeys] = useState(getDefaultActiveKeys());

  // Update active keys when preset or config changes
  useEffect(() => {
    setActiveKeys(getDefaultActiveKeys());
  }, [selectedPreset, config.num_of_malicious_clients, config.aggregation_strategy_keyword]);

  // Handle Esc key to navigate back to dashboard
  useEffect(() => {
    const handleEscape = (e) => {
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
      const attackName = config.attack_type === "gaussian_noise" ? "Gaussian noise" : "label flipping";
      parts.push(`tests ${attackName} attacks with ${config.aggregation_strategy_keyword} defense`);
    } else {
      parts.push(`runs a benign federated learning simulation using ${config.aggregation_strategy_keyword} aggregation`);
    }

    parts.push(`over ${config.num_of_rounds} round${config.num_of_rounds !== 1 ? 's' : ''} using ${config.num_of_clients} client${config.num_of_clients !== 1 ? 's' : ''}`);

    if (config.num_of_malicious_clients > 0) {
      parts.push(`(${config.num_of_malicious_clients} malicious)`);
    }

    parts.push(`on ${config.dataset_keyword} dataset`);

    if (config.remove_clients === "true" && config.begin_removing_from_round) {
      parts.push(`Defense will begin removing suspicious clients from round ${config.begin_removing_from_round}`);
    }

    return "This simulation " + parts.join(' ') + ".";
  };

  // Estimate resource usage
  const estimateResources = () => {
    const totalOperations = config.num_of_rounds * config.num_of_clients * config.num_of_client_epochs;
    const estimatedMinutes = Math.ceil(totalOperations / 10); // rough estimate: 10 ops/min on CPU
    const estimatedMemoryMB = config.num_of_clients * 50; // rough estimate: 50MB per client
    const estimatedDiskMB = config.save_plots === "true" || config.save_csv === "true" ? 10 : 2;

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
  const handleImportConfig = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
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
          {draftSaved && (
            <span className="badge bg-success">Draft saved</span>
          )}
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
            <Button variant="outline-secondary" size="sm" as="label" htmlFor="import-config" style={{ cursor: 'pointer', marginBottom: 0 }}>
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
                  backgroundColor: selectedPreset === key ? 'rgba(13, 110, 253, 0.1)' : ''
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
                backgroundColor: selectedPreset === null ? 'rgba(13, 110, 253, 0.1)' : ''
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
                <p className="small mb-2">Configure all parameters manually for specialized experiments</p>
                <div className="text-muted small">
                  <strong>Est. time:</strong> Varies
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </div>

      <Form onSubmit={handleSubmit}>
        <Accordion activeKey={activeKeys} onSelect={(keys) => setActiveKeys(keys)} alwaysOpen>
          <Accordion.Item eventKey="0">
            <Accordion.Header>
              <span className={sectionValidation.common.valid ? 'text-success' : 'text-danger'} style={{ marginRight: '8px' }}>
                {sectionValidation.common.valid ? '✓' : '✗'}
              </span>
              Common Settings <span className="text-muted ms-2" style={{ fontSize: '0.9em' }}>(6 required, 5 optional)</span>
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>
                  Strategy <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Aggregation algorithm for combining client updates. fedavg is simplest (average), others provide Byzantine robustness.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select name="aggregation_strategy_keyword" value={config.aggregation_strategy_keyword} onChange={handleChange}>
                  {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Dataset <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Training dataset for federated learning. Medical datasets (pneumoniamnist, bloodmnist) are common for healthcare FL.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Select name="dataset_keyword" value={config.dataset_keyword} onChange={handleChange}>
                  {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Model Type</Form.Label>
                <Form.Select name="model_type" value={config.model_type} onChange={handleChange}>
                  <option value="cnn">CNN</option>
                  <option value="transformer">Transformer</option>
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Use LLM"
                  name="use_llm"
                  checked={config.use_llm === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, use_llm: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              {needsLlmParams && (
                <>
                  <Form.Group className="mb-3">
                    <Form.Label>LLM Model</Form.Label>
                    <Form.Select name="llm_model" value={config.llm_model} onChange={handleChange}>
                      <option value="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext">BiomedBERT</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>LLM Fine-tuning</Form.Label>
                    <Form.Select name="llm_finetuning" value={config.llm_finetuning} onChange={handleChange}>
                      <option value="full">Full</option>
                      <option value="lora">LoRA</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>LLM Task</Form.Label>
                    <Form.Select name="llm_task" value={config.llm_task} onChange={handleChange}>
                      <option value="mlm">MLM (Masked Language Modeling)</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>LLM Chunk Size</Form.Label>
                    <Form.Control type="number" name="llm_chunk_size" value={config.llm_chunk_size} onChange={handleChange} />
                  </Form.Group>

                  {needsMLMParams && (
                    <Form.Group className="mb-3">
                      <Form.Label>MLM Probability</Form.Label>
                      <Form.Control type="number" step="0.01" name="mlm_probability" value={config.mlm_probability} onChange={handleChange} />
                    </Form.Group>
                  )}
                </>
              )}

              <Form.Group className="mb-3">
                <Form.Label>
                  Number of Rounds <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Communication rounds between server and clients. Start with 2-5 for quick tests, use 10+ for real experiments.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control type="number" name="num_of_rounds" value={config.num_of_rounds} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Number of Clients <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Total participating devices/clients. More clients = more realistic but slower simulation.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control type="number" name="num_of_clients" value={config.num_of_clients} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Batch Size <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Number of samples per training batch. Larger = faster but more memory. 32 is standard.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control type="number" name="batch_size" value={config.batch_size} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>
                  Client Epochs <span className="text-danger">*</span>{' '}
                  <OverlayTrigger
                    placement="right"
                    overlay={<Tooltip>Training passes each client performs locally before sending updates. 1 epoch is fastest.</Tooltip>}
                  >
                    <span style={{ cursor: 'help' }}>ℹ️</span>
                  </OverlayTrigger>
                </Form.Label>
                <Form.Control type="number" name="num_of_client_epochs" value={config.num_of_client_epochs} onChange={handleChange} />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="1">
            <Accordion.Header>
              <span className={sectionValidation.attack.valid ? 'text-success' : 'text-danger'} style={{ marginRight: '8px' }}>
                {sectionValidation.attack.valid ? '✓' : '✗'}
              </span>
              Attack Configuration
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>Number of Malicious Clients</Form.Label>
                <Form.Control type="number" name="num_of_malicious_clients" value={config.num_of_malicious_clients} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Attack Type</Form.Label>
                <Form.Select name="attack_type" value={config.attack_type} onChange={handleChange}>
                  {ATTACKS.map(a => <option key={a} value={a}>{a}</option>)}
                </Form.Select>
              </Form.Group>

              {needsGaussianParams && (
                <>
                  <Form.Group className="mb-3">
                    <Form.Label>Gaussian Noise Mean</Form.Label>
                    <Form.Control type="number" name="gaussian_noise_mean" value={config.gaussian_noise_mean || 0} onChange={handleChange} />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Gaussian Noise Std</Form.Label>
                    <Form.Control type="number" name="gaussian_noise_std" value={config.gaussian_noise_std || 1} onChange={handleChange} />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Attack Ratio</Form.Label>
                    <Form.Control type="number" step="0.01" name="attack_ratio" value={config.attack_ratio || 0.5} onChange={handleChange} />
                  </Form.Group>
                </>
              )}
            </Accordion.Body>
          </Accordion.Item>

          {(needsTrustParams || needsPidParams || needsKrumParams || needsTrimmedMeanParams) && (
            <Accordion.Item eventKey="2">
              <Accordion.Header>
                <span className={sectionValidation.strategy.valid ? 'text-success' : 'text-danger'} style={{ marginRight: '8px' }}>
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
                          overlay={<Tooltip>Round number when trust-based client filtering starts. Earlier = more aggressive filtering.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" name="begin_removing_from_round" value={config.begin_removing_from_round || 1} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Trust Threshold{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>Minimum trust score (0-1) for client inclusion. Lower = more permissive, higher = stricter filtering.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.01" name="trust_threshold" value={config.trust_threshold || 0.5} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Beta Value{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>Exponential decay factor for trust score updates. Higher (closer to 1) = trust changes slowly.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.01" name="beta_value" value={config.beta_value || 0.9} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Number of Clusters{' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>Number of client clusters for trust grouping. More clusters = finer-grained trust analysis.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" name="num_of_clusters" value={config.num_of_clusters || 1} onChange={handleChange} />
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
                          overlay={<Tooltip>Threshold for outlier detection. Updates beyond this many standard deviations are filtered.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.1" name="num_std_dev" value={config.num_std_dev || 2.0} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Kp (Proportional Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>PID controller proportional term. Controls reaction to current error. Higher = more aggressive correction.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.01" name="Kp" value={config.Kp || 1.0} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Ki (Integral Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>PID controller integral term. Eliminates steady-state error by accumulating past errors.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.01" name="Ki" value={config.Ki || 0.1} onChange={handleChange} />
                    </Form.Group>

                    <Form.Group className="mb-3">
                      <Form.Label>
                        Kd (Derivative Gain){' '}
                        <OverlayTrigger
                          placement="right"
                          overlay={<Tooltip>PID controller derivative term. Predicts future error based on rate of change. Reduces overshoot.</Tooltip>}
                        >
                          <span style={{ cursor: 'help' }}>ℹ️</span>
                        </OverlayTrigger>
                      </Form.Label>
                      <Form.Control type="number" step="0.01" name="Kd" value={config.Kd || 0.01} onChange={handleChange} />
                    </Form.Group>
                  </>
                )}

                {needsKrumParams && (
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Krum Selections{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={<Tooltip>Number of closest clients to aggregate. Lower = more Byzantine robustness but less data diversity.</Tooltip>}
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control type="number" name="num_krum_selections" value={config.num_krum_selections || 5} onChange={handleChange} />
                  </Form.Group>
                )}

                {needsTrimmedMeanParams && (
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Trim Ratio{' '}
                      <OverlayTrigger
                        placement="right"
                        overlay={<Tooltip>Fraction of extreme values to remove from both ends (0-0.5). Higher = more aggressive outlier filtering.</Tooltip>}
                      >
                        <span style={{ cursor: 'help' }}>ℹ️</span>
                      </OverlayTrigger>
                    </Form.Label>
                    <Form.Control type="number" step="0.01" name="trim_ratio" value={config.trim_ratio || 0.1} onChange={handleChange} />
                  </Form.Group>
                )}
              </Accordion.Body>
            </Accordion.Item>
          )}

          <Accordion.Item eventKey="3">
            <Accordion.Header>
              <span className={sectionValidation.resources.valid ? 'text-success' : 'text-danger'} style={{ marginRight: '8px' }}>
                {sectionValidation.resources.valid ? '✓' : '✗'}
              </span>
              Resource & Output Settings
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>Training Device</Form.Label>
                <Form.Select name="training_device" value={config.training_device} onChange={handleChange}>
                  {DEVICES.map(d => <option key={d} value={d}>{d}</option>)}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>CPUs per Client</Form.Label>
                <Form.Control type="number" name="cpus_per_client" value={config.cpus_per_client} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>GPUs per Client</Form.Label>
                <Form.Control type="number" step="0.1" name="gpus_per_client" value={config.gpus_per_client} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Training Subset Fraction</Form.Label>
                <Form.Control type="number" step="0.1" name="training_subset_fraction" value={config.training_subset_fraction} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Show Plots"
                  name="show_plots"
                  checked={config.show_plots === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, show_plots: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Save Plots"
                  name="save_plots"
                  checked={config.save_plots === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, save_plots: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Save CSV"
                  name="save_csv"
                  checked={config.save_csv === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, save_csv: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Preserve Dataset"
                  name="preserve_dataset"
                  checked={config.preserve_dataset === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, preserve_dataset: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Remove Clients"
                  name="remove_clients"
                  checked={config.remove_clients === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, remove_clients: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Check
                  type="switch"
                  label="Strict Mode"
                  name="strict_mode"
                  checked={config.strict_mode === "true"}
                  onChange={(e) => setConfig(prev => ({ ...prev, strict_mode: e.target.checked ? "true" : "false" }))}
                />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="4">
            <Accordion.Header>
              <span className={sectionValidation.flower.valid ? 'text-success' : 'text-danger'} style={{ marginRight: '8px' }}>
                {sectionValidation.flower.valid ? '✓' : '✗'}
              </span>
              Flower Framework Settings
            </Accordion.Header>
            <Accordion.Body>
              <Form.Group className="mb-3">
                <Form.Label>Min Fit Clients</Form.Label>
                <Form.Control type="number" name="min_fit_clients" value={config.min_fit_clients} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Min Evaluate Clients</Form.Label>
                <Form.Control type="number" name="min_evaluate_clients" value={config.min_evaluate_clients} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Min Available Clients</Form.Label>
                <Form.Control type="number" name="min_available_clients" value={config.min_available_clients} onChange={handleChange} />
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Evaluate Metrics Aggregation Function</Form.Label>
                <Form.Control type="text" name="evaluate_metrics_aggregation_fn" value={config.evaluate_metrics_aggregation_fn} onChange={handleChange} />
              </Form.Group>
            </Accordion.Body>
          </Accordion.Item>
        </Accordion>

        {isValid && (
          <Card className="mt-4 bg-light border-primary">
            <Card.Body>
              <h6 className="mb-3">📋 Configuration Summary</h6>
              <p className="mb-3 text-muted">{generateSummary()}</p>

              <Row className="g-3">
                <Col md={4}>
                  <div className="small">
                    <strong>⏱️ Estimated Time:</strong>{' '}
                    <span className={isNonDefault('num_of_rounds', config.num_of_rounds) || isNonDefault('num_of_clients', config.num_of_clients) ? 'fw-bold text-primary' : ''}>
                      ~{estimateResources().estimatedMinutes} min
                    </span>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="small">
                    <strong>💾 Memory Usage:</strong>{' '}
                    <span className={isNonDefault('num_of_clients', config.num_of_clients) ? 'fw-bold text-primary' : ''}>
                      ~{estimateResources().estimatedMemoryMB} MB
                    </span>
                  </div>
                </Col>
                <Col md={4}>
                  <div className="small">
                    <strong>💿 Disk Space:</strong>{' '}
                    <span className={isNonDefault('save_plots', config.save_plots) || isNonDefault('save_csv', config.save_csv) ? 'fw-bold text-primary' : ''}>
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
                        isNonDefault('aggregation_strategy_keyword', config.aggregation_strategy_keyword) && `Strategy (${config.aggregation_strategy_keyword})`,
                        isNonDefault('dataset_keyword', config.dataset_keyword) && `Dataset (${config.dataset_keyword})`,
                        isNonDefault('num_of_rounds', config.num_of_rounds) && `Rounds (${config.num_of_rounds})`,
                        isNonDefault('num_of_clients', config.num_of_clients) && `Clients (${config.num_of_clients})`,
                      ].filter(Boolean).join(', ')}
                    </div>
                  </div>
                )}
            </Card.Body>
          </Card>
        )}

        <div className="mt-3">
          <div className="d-flex align-items-center gap-3">
            <Button variant="primary" type="submit" disabled={submitting || !isValid}>
              {submitting ? 'Launching...' : 'Launch Simulation'}
            </Button>
            {!isValid && totalIssues > 0 && (
              <span className="badge bg-danger">
                {totalIssues} issue{totalIssues !== 1 ? 's' : ''} remaining
              </span>
            )}
          </div>
          {!isValid && <div className="text-muted mt-2">Please complete all required fields to launch simulation</div>}
        </div>

        {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
      </Form>
    </div>
  );
}

export default NewSimulation;