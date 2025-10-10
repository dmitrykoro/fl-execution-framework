import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  Spinner,
  Alert,
  Card,
  Tabs,
  Tab,
  Table,
  ProgressBar,
  ListGroup,
  Accordion,
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap';
import useApi from '../hooks/useApi';
import { getSimulationDetails, getSimulationStatus, getResultFile, createSimulation } from '../api';
import InteractivePlots from './InteractivePlots';
import OutlineButton from './common/Button/OutlineButton';
import { MaterialIcon } from './common/Icon/MaterialIcon';

function SimulationDetails() {
  const { simulationId } = useParams();
  const navigate = useNavigate();
  const { data: details, loading, error, refetch } = useApi(getSimulationDetails, simulationId);
  const [csvData, setCsvData] = useState({});
  const [isCloning, setIsCloning] = useState(false);
  const [currentStatus, setCurrentStatus] = useState(null);

  // Poll for status updates when simulation is running
  useEffect(() => {
    if (!details) return;

    const pollStatus = async () => {
      try {
        const response = await getSimulationStatus(simulationId);
        setCurrentStatus(response.data.status);

        // Refetch details when status changes to completed
        if (response.data.status === 'completed' && details.status !== 'completed') {
          refetch();
        }
      } catch (err) {
        console.error('Failed to poll status:', err);
      }
    };

    // Poll every 2 seconds if running
    if (details.status === 'running') {
      pollStatus();
      const interval = setInterval(pollStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [details, simulationId, refetch]);

  useEffect(() => {
    if (details && details.result_files) {
      const csvFiles = details.result_files.filter(file => file.endsWith('.csv'));
      csvFiles.forEach(async file => {
        try {
          const response = await getResultFile(simulationId, file);
          setCsvData(prev => ({ ...prev, [file]: response.data }));
        } catch (err) {
          console.error(`Failed to load CSV ${file}:`, err);
        }
      });
    }
  }, [details, simulationId]);

  if (loading) {
    return (
      <div>
        <h1>Simulation Details: {simulationId}</h1>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>Simulation Details: {simulationId}</h1>
        <Alert variant="danger">{error}</Alert>
      </div>
    );
  }

  if (!details) {
    return null;
  }

  const { config, result_files, status } = details;
  const displayStatus = currentStatus || status;
  const cfg = config.shared_settings || config;
  const csvFiles = result_files.filter(file => file.endsWith('.csv'));

  const statusVariant =
    displayStatus === 'completed'
      ? 'success'
      : displayStatus === 'failed'
        ? 'danger'
        : displayStatus === 'running'
          ? 'primary'
          : 'secondary';

  const handleRunAgain = async () => {
    setIsCloning(true);
    try {
      // Unwrap shared_settings if present, since API will wrap it again
      const configToSend = config.shared_settings || config;
      const response = await createSimulation(configToSend);
      navigate(`/simulations/${response.data.simulation_id}`);
    } catch (err) {
      console.error('Failed to clone simulation:', err);
      alert('Failed to start new simulation');
      setIsCloning(false);
    }
  };

  // Generate educational insights from CSV data
  const generateInsights = () => {
    const insights = [];

    // Get round metrics
    const roundMetrics = csvData['csv/round_metrics_0.csv'];
    const perClientMetrics = csvData['csv/per_client_metrics_0.csv'];

    if (!roundMetrics || roundMetrics.length === 0) {
      return insights;
    }

    // Analyze accuracy improvement
    if (roundMetrics.length >= 2) {
      const firstAccuracy = parseFloat(roundMetrics[0].average_accuracy_history);
      const lastAccuracy = parseFloat(
        roundMetrics[roundMetrics.length - 1].average_accuracy_history
      );
      const improvement = (((lastAccuracy - firstAccuracy) / firstAccuracy) * 100).toFixed(1);

      if (improvement > 0) {
        insights.push({
          type: 'success',
          icon: '📈',
          text: `Model accuracy improved by ${improvement}% over ${roundMetrics.length} rounds (from ${(firstAccuracy * 100).toFixed(1)}% to ${(lastAccuracy * 100).toFixed(1)}%)`,
        });
      } else if (improvement < 0) {
        insights.push({
          type: 'warning',
          icon: '⚠️',
          text: `Model accuracy decreased by ${Math.abs(improvement)}% - this may indicate attack or poor hyperparameters`,
        });
      }
    }

    // Analyze malicious clients
    if (cfg.num_of_malicious_clients > 0) {
      insights.push({
        type: 'info',
        icon: '🎯',
        text: `Simulation includes ${cfg.num_of_malicious_clients} malicious client(s) using ${cfg.attack_type} attack`,
      });

      // Check removal metrics
      if (cfg.remove_clients === 'true' && roundMetrics.length > 0) {
        const lastRound = roundMetrics[roundMetrics.length - 1];
        const removalAccuracy = parseFloat(lastRound.removal_accuracy_history);
        const removalPrecision = parseFloat(lastRound.removal_precision_history);
        const removalRecall = parseFloat(lastRound.removal_recall_history);

        if (removalAccuracy === 1.0) {
          insights.push({
            type: 'success',
            icon: '✓',
            text: `Defense strategy (${cfg.aggregation_strategy_keyword}) successfully identified all malicious clients with 100% accuracy`,
          });
        } else if (removalAccuracy >= 0.7) {
          insights.push({
            type: 'success',
            icon: '✓',
            text: `Defense detected malicious clients with ${(removalAccuracy * 100).toFixed(0)}% accuracy (Precision: ${(removalPrecision * 100).toFixed(0)}%, Recall: ${(removalRecall * 100).toFixed(0)}%)`,
          });
        } else if (removalAccuracy > 0) {
          insights.push({
            type: 'warning',
            icon: '⚠️',
            text: `Defense partially effective: ${(removalAccuracy * 100).toFixed(0)}% accuracy in detecting malicious clients`,
          });
        }
      }
    } else {
      insights.push({
        type: 'info',
        icon: 'ℹ️',
        text: 'Baseline simulation with no malicious clients - observing natural federated learning behavior',
      });
    }

    // Analyze client participation (from per_client_metrics)
    if (perClientMetrics && perClientMetrics.length > 0) {
      const lastRound = perClientMetrics[perClientMetrics.length - 1];
      const participationKeys = Object.keys(lastRound).filter(k =>
        k.includes('aggregation_participation_history')
      );
      const activeClients = participationKeys.filter(
        k => lastRound[k] === '1' || lastRound[k] === 1
      ).length;
      const removedClients = cfg.num_of_clients - activeClients;

      if (removedClients > 0) {
        insights.push({
          type: 'info',
          icon: '🔒',
          text: `${removedClients} client(s) removed from aggregation by round ${roundMetrics.length} (${activeClients} active clients remaining)`,
        });
      }
    }

    // Analyze defense strategy behavior
    if (cfg.aggregation_strategy_keyword === 'pid' && cfg.remove_clients === 'true') {
      const beginRemoving = cfg.begin_removing_from_round || 2;
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `PID-based removal strategy started evaluating clients from round ${beginRemoving} with ${cfg.pid_p || 0.1} proportional gain`,
      });
    } else if (cfg.aggregation_strategy_keyword === 'krum') {
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `Krum aggregation selects the most trustworthy client update based on distance metrics`,
      });
    } else if (cfg.aggregation_strategy_keyword === 'trimmed_mean') {
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `Trimmed mean removes extreme updates before aggregation for robustness`,
      });
    }

    // Dataset and model info
    insights.push({
      type: 'info',
      icon: '📊',
      text: `Trained ${cfg.model_type || 'cnn'} model on ${cfg.dataset_keyword} dataset with ${cfg.num_of_clients} clients`,
    });

    return insights;
  };

  const insights = displayStatus === 'completed' ? generateInsights() : [];

  // Helper function to copy CSV data to clipboard
  const copyCSVToClipboard = (jsonData, filename) => {
    try {
      if (!jsonData || jsonData.length === 0) {
        alert('No data to copy');
        return;
      }

      // Convert JSON to CSV string
      const columns = Object.keys(jsonData[0]);
      const header = columns.join(',');
      const rows = jsonData.map(row =>
        columns
          .map(col => {
            const value = row[col];
            // Escape values containing commas or quotes
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value;
          })
          .join(',')
      );
      const csvString = [header, ...rows].join('\n');

      // Copy to clipboard
      navigator.clipboard.writeText(csvString).then(
        () => {
          alert(`✓ Copied ${filename} to clipboard!`);
        },
        () => {
          alert('Failed to copy to clipboard. Please try again.');
        }
      );
    } catch (error) {
      console.error('Error copying to clipboard:', error);
      alert('Failed to copy to clipboard. Please try again.');
    }
  };

  // Human-readable config display component
  const ConfigDisplay = ({ config }) => {
    const [showRawJSON, setShowRawJSON] = useState(false);
    const cfg = config.shared_settings || config;

    const ConfigRow = ({ label, value, tooltip }) => (
      <tr>
        <td className="fw-semibold" style={{ width: '40%' }}>
          {label}
          {tooltip && (
            <OverlayTrigger placement="right" overlay={<Tooltip>{tooltip}</Tooltip>}>
              <span style={{ cursor: 'help', marginLeft: '4px', fontSize: '0.9rem' }}>ℹ️</span>
            </OverlayTrigger>
          )}
        </td>
        <td>{value}</td>
      </tr>
    );

    const formatDatasetName = keyword => {
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

    const formatStrategyName = keyword => {
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

    const formatAttackName = keyword => {
      const names = {
        gaussian_noise: 'Gaussian Noise',
        label_flipping: 'Label Flipping',
      };
      return names[keyword] || keyword;
    };

    if (showRawJSON) {
      return (
        <>
          <div className="d-flex justify-content-between align-items-center mb-3">
            <h5 className="mb-0">📋 Raw Configuration JSON</h5>
            <OutlineButton onClick={() => setShowRawJSON(false)}>View Human-Readable</OutlineButton>
          </div>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              color: '#E6E1E5', // Light text for dark mode
            }}
          >
            {JSON.stringify(config, null, 2)}
          </pre>
        </>
      );
    }

    return (
      <>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5 className="mb-0">📋 Configuration</h5>
          <OutlineButton onClick={() => setShowRawJSON(true)}>View Raw JSON</OutlineButton>
        </div>

        <Accordion defaultActiveKey="0">
          <Accordion.Item eventKey="0">
            <Accordion.Header>Common Settings</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  <ConfigRow
                    label="Aggregation Strategy"
                    value={formatStrategyName(cfg.aggregation_strategy_keyword)}
                    tooltip="Aggregation algorithm for combining client updates. fedavg is simplest (average), others provide Byzantine robustness."
                  />
                  <ConfigRow
                    label="Dataset"
                    value={formatDatasetName(cfg.dataset_keyword)}
                    tooltip="Training dataset for federated learning. Medical datasets (pneumoniamnist, bloodmnist) are common for healthcare FL."
                  />
                  <ConfigRow
                    label="Model Type"
                    value={cfg.model_type === 'cnn' ? 'CNN' : cfg.model_type || 'CNN'}
                  />
                  <ConfigRow label="Use LLM" value={cfg.use_llm === 'true' ? 'Yes' : 'No'} />
                  <ConfigRow
                    label="Number of Rounds"
                    value={cfg.num_of_rounds}
                    tooltip="Communication rounds between server and clients. Start with 2-5 for quick tests, use 10+ for real experiments."
                  />
                  <ConfigRow
                    label="Number of Clients"
                    value={cfg.num_of_clients}
                    tooltip="Total participating devices/clients. More clients = more realistic but slower simulation."
                  />
                  <ConfigRow
                    label="Batch Size"
                    value={cfg.batch_size}
                    tooltip="Number of samples per training batch. Larger = faster but more memory. 32 is standard."
                  />
                  <ConfigRow
                    label="Client Epochs"
                    value={cfg.num_of_client_epochs}
                    tooltip="Training passes each client performs locally before sending updates. 1 epoch is fastest."
                  />
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="1">
            <Accordion.Header>Attack Configuration</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  <ConfigRow label="Malicious Clients" value={cfg.num_of_malicious_clients || 0} />
                  <ConfigRow label="Attack Type" value={formatAttackName(cfg.attack_type)} />
                  {cfg.attack_type === 'gaussian_noise' && (
                    <>
                      <ConfigRow label="Gaussian Noise Mean" value={cfg.gaussian_noise_mean ?? 0} />
                      <ConfigRow label="Gaussian Noise Std" value={cfg.gaussian_noise_std ?? 1} />
                      <ConfigRow label="Attack Ratio" value={cfg.attack_ratio ?? 0.5} />
                    </>
                  )}
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="2">
            <Accordion.Header>Strategy-Specific Parameters</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  {cfg.aggregation_strategy_keyword === 'trust' && (
                    <>
                      <ConfigRow
                        label="Begin Removing From Round"
                        value={cfg.begin_removing_from_round || 1}
                        tooltip="Round number when trust-based client filtering starts. Earlier = more aggressive filtering."
                      />
                      <ConfigRow
                        label="Trust Threshold"
                        value={cfg.trust_threshold || 0.5}
                        tooltip="Minimum trust score (0-1) for client inclusion. Lower = more permissive, higher = stricter filtering."
                      />
                      <ConfigRow
                        label="Beta Value"
                        value={cfg.beta_value || 0.9}
                        tooltip="Exponential decay factor for trust score updates. Higher (closer to 1) = trust changes slowly."
                      />
                      <ConfigRow
                        label="Number of Clusters"
                        value={cfg.num_of_clusters || 1}
                        tooltip="Number of client clusters for trust grouping. More clusters = finer-grained trust analysis."
                      />
                    </>
                  )}
                  {['pid', 'pid_scaled', 'pid_standardized'].includes(
                    cfg.aggregation_strategy_keyword
                  ) && (
                    <>
                      <ConfigRow
                        label="Number of Std Deviations"
                        value={cfg.num_std_dev || 2.0}
                        tooltip="Threshold for outlier detection. Updates beyond this many standard deviations are filtered."
                      />
                      <ConfigRow
                        label="Kp (Proportional Gain)"
                        value={cfg.Kp || 1.0}
                        tooltip="PID controller proportional term. Controls reaction to current error. Higher = more aggressive correction."
                      />
                      <ConfigRow
                        label="Ki (Integral Gain)"
                        value={cfg.Ki || 0.1}
                        tooltip="PID controller integral term. Eliminates steady-state error by accumulating past errors."
                      />
                      <ConfigRow
                        label="Kd (Derivative Gain)"
                        value={cfg.Kd || 0.01}
                        tooltip="PID controller derivative term. Predicts future error based on rate of change. Reduces overshoot."
                      />
                    </>
                  )}
                  {['multi-krum', 'krum', 'multi-krum-based'].includes(
                    cfg.aggregation_strategy_keyword
                  ) && (
                    <ConfigRow
                      label="Krum Selections"
                      value={cfg.num_krum_selections || 5}
                      tooltip="Number of closest clients to aggregate. Lower = more Byzantine robustness but less data diversity."
                    />
                  )}
                  {cfg.aggregation_strategy_keyword === 'trimmed_mean' && (
                    <ConfigRow
                      label="Trim Ratio"
                      value={cfg.trim_ratio || 0.1}
                      tooltip="Fraction of extreme values to remove from both ends (0-0.5). Higher = more aggressive outlier filtering."
                    />
                  )}
                  {![
                    'trust',
                    'pid',
                    'pid_scaled',
                    'pid_standardized',
                    'multi-krum',
                    'krum',
                    'multi-krum-based',
                    'trimmed_mean',
                  ].includes(cfg.aggregation_strategy_keyword) && (
                    <tr>
                      <td colSpan="2" className="text-muted fst-italic">
                        No strategy-specific parameters for{' '}
                        {formatStrategyName(cfg.aggregation_strategy_keyword)}
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="3">
            <Accordion.Header>Resource & Output Settings</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  <ConfigRow
                    label="Training Device"
                    value={cfg.training_device?.toUpperCase() || 'CPU'}
                  />
                  <ConfigRow label="CPUs per Client" value={cfg.cpus_per_client || 1} />
                  <ConfigRow label="GPUs per Client" value={cfg.gpus_per_client || 0.0} />
                  <ConfigRow
                    label="Training Subset Fraction"
                    value={cfg.training_subset_fraction || 0.9}
                  />
                  <ConfigRow label="Show Plots" value={cfg.show_plots === 'true' ? 'Yes' : 'No'} />
                  <ConfigRow label="Save Plots" value={cfg.save_plots === 'true' ? 'Yes' : 'No'} />
                  <ConfigRow label="Save CSV" value={cfg.save_csv === 'true' ? 'Yes' : 'No'} />
                  <ConfigRow
                    label="Preserve Dataset"
                    value={cfg.preserve_dataset === 'true' ? 'Yes' : 'No'}
                  />
                  <ConfigRow
                    label="Remove Clients"
                    value={cfg.remove_clients === 'true' ? 'Yes' : 'No'}
                  />
                  <ConfigRow
                    label="Strict Mode"
                    value={cfg.strict_mode === 'true' ? 'Yes' : 'No'}
                  />
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>

          <Accordion.Item eventKey="4">
            <Accordion.Header>Flower Framework Settings</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  <ConfigRow label="Min Fit Clients" value={cfg.min_fit_clients || 5} />
                  <ConfigRow label="Min Evaluate Clients" value={cfg.min_evaluate_clients || 5} />
                  <ConfigRow label="Min Available Clients" value={cfg.min_available_clients || 5} />
                  <ConfigRow
                    label="Evaluate Metrics Aggregation"
                    value={cfg.evaluate_metrics_aggregation_fn || 'weighted_average'}
                  />
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>

          {cfg.use_llm === 'true' && (
            <Accordion.Item eventKey="5">
              <Accordion.Header>LLM Settings</Accordion.Header>
              <Accordion.Body>
                <Table size="sm" className="mb-0">
                  <tbody>
                    <ConfigRow label="LLM Model" value={cfg.llm_model || 'BiomedBERT'} />
                    <ConfigRow
                      label="Fine-tuning Method"
                      value={cfg.llm_finetuning === 'lora' ? 'LoRA' : 'Full'}
                    />
                    <ConfigRow
                      label="Task"
                      value={
                        cfg.llm_task === 'mlm' ? 'MLM (Masked Language Modeling)' : cfg.llm_task
                      }
                    />
                    <ConfigRow label="Chunk Size" value={cfg.llm_chunk_size || 256} />
                    {cfg.llm_task === 'mlm' && (
                      <ConfigRow label="MLM Probability" value={cfg.mlm_probability || 0.15} />
                    )}
                    {cfg.llm_finetuning === 'lora' && (
                      <>
                        <ConfigRow label="LoRA Rank" value={cfg.lora_rank || 16} />
                        <ConfigRow label="LoRA Alpha" value={cfg.lora_alpha || 32} />
                        <ConfigRow label="LoRA Dropout" value={cfg.lora_dropout || 0.1} />
                        <ConfigRow
                          label="LoRA Target Modules"
                          value={
                            Array.isArray(cfg.lora_target_modules)
                              ? cfg.lora_target_modules.join(', ')
                              : 'query, value'
                          }
                        />
                      </>
                    )}
                  </tbody>
                </Table>
              </Accordion.Body>
            </Accordion.Item>
          )}
        </Accordion>
      </>
    );
  };

  const displayName = cfg.display_name;

  return (
    <div>
      <div className="d-flex flex-column gap-2 mb-3">
        <div className="d-flex flex-column flex-md-row align-items-stretch align-items-md-center gap-2 gap-md-3">
          <div className="d-flex align-items-center gap-2 flex-wrap flex-grow-1">
            <h4 className="mb-0">{displayName || simulationId}</h4>
            <span className={`badge bg-${statusVariant}`}>{displayStatus}</span>
          </div>
          <OutlineButton
            variant="outline-primary"
            onClick={handleRunAgain}
            disabled={isCloning}
            className="flex-shrink-0 d-flex align-items-center gap-2"
          >
            <MaterialIcon name="replay" size={20} />
            {isCloning ? 'Starting...' : 'Run Again'}
          </OutlineButton>
        </div>
        {displayName && (
          <div className="text-muted small">
            ID: <code>{simulationId}</code>
          </div>
        )}
      </div>

      {displayStatus === 'running' && (
        <Card className="mb-3">
          <Card.Body>
            <div className="d-flex align-items-center gap-3 mb-2">
              <h6 className="mb-0">Simulation in Progress</h6>
            </div>
            <ProgressBar animated now={100} variant="primary" className="mb-2" />
            <div className="text-muted small">
              <p className="mb-1">
                Running federated learning simulation with {cfg.num_of_clients} clients over{' '}
                {cfg.num_of_rounds} rounds...
              </p>
              <p className="mb-0">
                Status updates every 2 seconds. Results will appear automatically when complete.
              </p>
            </div>
          </Card.Body>
        </Card>
      )}

      {displayStatus === 'completed' && (
        <Alert variant="success" className="mb-3">
          <strong>✓ Simulation completed successfully!</strong> View results in the tabs below.
        </Alert>
      )}

      {displayStatus === 'failed' && (
        <Alert variant="danger" className="mb-3">
          <strong>✗ Simulation failed.</strong> Check the error logs for details.
        </Alert>
      )}

      <Tabs defaultActiveKey="insights" className="mt-4 flex-nowrap overflow-auto">
        <Tab eventKey="insights" title="Insights">
          <Card className="mt-3">
            <Card.Body>
              {insights.length > 0 ? (
                <>
                  <h5 className="mb-3">📚 Educational Insights</h5>
                  <p className="text-muted mb-3">
                    Automatic analysis of simulation results to help understand federated learning
                    behavior and defense effectiveness.
                  </p>
                  <ListGroup>
                    {insights.map((insight, idx) => (
                      <ListGroup.Item
                        key={idx}
                        variant={
                          insight.type === 'success'
                            ? 'success'
                            : insight.type === 'warning'
                              ? 'warning'
                              : 'light'
                        }
                        className="d-flex align-items-start gap-2"
                      >
                        <span style={{ fontSize: '1.2rem', minWidth: '24px' }}>{insight.icon}</span>
                        <span>{insight.text}</span>
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                </>
              ) : (
                <Alert variant="info">
                  Insights will be generated once the simulation completes and metrics are
                  available.
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="plots" title="Plots">
          <InteractivePlots simulation={{ ...details, id: simulationId }} />
        </Tab>

        <Tab eventKey="metrics" title="Metrics">
          <div className="mt-3">
            {csvFiles.length > 0 ? (
              <>
                {/* Render Round-Level Summary */}
                {csvData['csv/round_metrics_0.csv'] && (
                  <Card className="mb-4">
                    <Card.Header>
                      <div className="d-flex justify-content-between align-items-center">
                        <h5 className="mb-0">📊 Round-by-Round Performance</h5>
                        <small className="text-muted">Key metrics for each training round</small>
                      </div>
                    </Card.Header>
                    <Card.Body>
                      <div style={{ overflowX: 'auto' }}>
                        <Table striped hover size="sm">
                          <thead>
                            <tr>
                              <th
                                style={{
                                  position: 'sticky',
                                  left: 0,
                                  backgroundColor: 'var(--bs-table-bg)',
                                  zIndex: 1,
                                }}
                              >
                                Round
                              </th>
                              <th>
                                <OverlayTrigger
                                  placement="top"
                                  overlay={
                                    <Tooltip>Average accuracy across all clients (0-100%)</Tooltip>
                                  }
                                >
                                  <span style={{ cursor: 'help' }}>Accuracy 📈</span>
                                </OverlayTrigger>
                              </th>
                              <th>
                                <OverlayTrigger
                                  placement="top"
                                  overlay={
                                    <Tooltip>
                                      Average loss across all clients (lower is better)
                                    </Tooltip>
                                  }
                                >
                                  <span style={{ cursor: 'help' }}>Loss 📉</span>
                                </OverlayTrigger>
                              </th>
                              {cfg.num_of_malicious_clients > 0 &&
                                cfg.remove_clients === 'true' && (
                                  <>
                                    <th>
                                      <OverlayTrigger
                                        placement="top"
                                        overlay={
                                          <Tooltip>
                                            Percentage of malicious clients correctly identified
                                          </Tooltip>
                                        }
                                      >
                                        <span style={{ cursor: 'help' }}>
                                          Detection Accuracy 🎯
                                        </span>
                                      </OverlayTrigger>
                                    </th>
                                    <th>
                                      <OverlayTrigger
                                        placement="top"
                                        overlay={
                                          <Tooltip>
                                            Of flagged clients, what % were actually malicious
                                          </Tooltip>
                                        }
                                      >
                                        <span style={{ cursor: 'help' }}>Precision ✓</span>
                                      </OverlayTrigger>
                                    </th>
                                    <th>
                                      <OverlayTrigger
                                        placement="top"
                                        overlay={
                                          <Tooltip>
                                            Of all malicious clients, what % were caught
                                          </Tooltip>
                                        }
                                      >
                                        <span style={{ cursor: 'help' }}>Recall 🔍</span>
                                      </OverlayTrigger>
                                    </th>
                                  </>
                                )}
                            </tr>
                          </thead>
                          <tbody>
                            {csvData['csv/round_metrics_0.csv'].map((row, idx) => {
                              const accuracy = parseFloat(row.average_accuracy_history || 0);
                              const loss = parseFloat(row.aggregated_loss_history || 0);
                              const detectionAcc = parseFloat(row.removal_accuracy_history || 0);
                              const precision = parseFloat(row.removal_precision_history || 0);
                              const recall = parseFloat(row.removal_recall_history || 0);

                              return (
                                <tr key={idx}>
                                  <td
                                    style={{
                                      position: 'sticky',
                                      left: 0,
                                      backgroundColor: 'var(--bs-table-bg)',
                                      fontWeight: 'bold',
                                    }}
                                  >
                                    {parseInt(row['round #'] || row.round || idx + 1)}
                                  </td>
                                  <td
                                    className={
                                      accuracy > 0.7
                                        ? 'text-success fw-semibold'
                                        : accuracy > 0.4
                                          ? 'text-warning'
                                          : 'text-danger'
                                    }
                                  >
                                    {(accuracy * 100).toFixed(1)}%
                                  </td>
                                  <td
                                    className={
                                      loss < 0.1 ? 'text-success' : loss < 0.5 ? 'text-warning' : ''
                                    }
                                  >
                                    {loss.toFixed(4)}
                                  </td>
                                  {cfg.num_of_malicious_clients > 0 &&
                                    cfg.remove_clients === 'true' && (
                                      <>
                                        <td
                                          className={
                                            detectionAcc === 1.0
                                              ? 'text-success fw-bold'
                                              : detectionAcc > 0.7
                                                ? 'text-success'
                                                : detectionAcc > 0
                                                  ? 'text-warning'
                                                  : ''
                                          }
                                        >
                                          {isNaN(detectionAcc) || detectionAcc === 0
                                            ? '—'
                                            : (detectionAcc * 100).toFixed(0) + '%'}
                                        </td>
                                        <td>
                                          {isNaN(precision) || precision === 0
                                            ? '—'
                                            : (precision * 100).toFixed(0) + '%'}
                                        </td>
                                        <td>
                                          {isNaN(recall) || recall === 0
                                            ? '—'
                                            : (recall * 100).toFixed(0) + '%'}
                                        </td>
                                      </>
                                    )}
                                </tr>
                              );
                            })}
                          </tbody>
                        </Table>
                      </div>
                      <div className="mt-3 small text-muted">
                        <strong>📖 How to interpret:</strong>
                        <div className="mb-0 mt-2">
                          <div>
                            <strong>Accuracy:</strong> Higher is better.{' '}
                            <span className="text-success">Green (&gt;70%)</span> = good,{' '}
                            <span className="text-warning">yellow (40-70%)</span> = learning,{' '}
                            <span className="text-danger">red (&lt;40%)</span> = poor
                          </div>
                          <div>
                            <strong>Loss:</strong> Lower is better. Shows how far predictions are
                            from true values
                          </div>
                          {cfg.num_of_malicious_clients > 0 && cfg.remove_clients === 'true' && (
                            <>
                              <div>
                                <strong>Detection Accuracy:</strong> How well the defense identifies
                                malicious clients. 100% = perfect detection!
                              </div>
                              <div>
                                <strong>Precision:</strong> Avoids false positives (flagging honest
                                clients as malicious)
                              </div>
                              <div>
                                <strong>Recall:</strong> Catches all actual malicious clients (no
                                false negatives)
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </Card.Body>
                  </Card>
                )}

                {/* Render Per-Client Summary (only show if interesting metrics exist) */}
                {csvData['csv/exec_stats_0.csv'] && (
                  <Card className="mb-4">
                    <Card.Header>
                      <div className="d-flex justify-content-between align-items-center">
                        <h5 className="mb-0">⏱️ Execution Statistics</h5>
                        <small className="text-muted">Performance timing breakdown</small>
                      </div>
                    </Card.Header>
                    <Card.Body>
                      {(() => {
                        const stats = csvData['csv/exec_stats_0.csv'][0];
                        const meanAccuracy = parseFloat(stats.mean_average_accuracy_history || 0);

                        return (
                          <>
                            <div className="row g-3">
                              <div className="col-12 col-sm-6 col-md-4">
                                <div className="p-3 border rounded">
                                  <div className="small text-muted">Final Accuracy</div>
                                  <div className="h4 mb-0 text-dark">
                                    {(meanAccuracy * 100).toFixed(1)}%
                                  </div>
                                </div>
                              </div>
                              <div className="col-12 col-sm-6 col-md-4">
                                <div className="p-3 border rounded">
                                  <div className="small text-muted">Total Rounds</div>
                                  <div className="h4 mb-0 text-dark">{cfg.num_of_rounds}</div>
                                </div>
                              </div>
                              <div className="col-12 col-sm-6 col-md-4">
                                <div className="p-3 border rounded">
                                  <div className="small text-muted">Total Clients</div>
                                  <div className="h4 mb-0 text-dark">{cfg.num_of_clients}</div>
                                </div>
                              </div>
                            </div>
                            <div className="mt-3 small text-muted">
                              <strong>💡 Tip:</strong> Compare these metrics across different
                              aggregation strategies to see which defends best against attacks!
                            </div>
                          </>
                        );
                      })()}
                    </Card.Body>
                  </Card>
                )}

                {/* Raw Data Accordion (for advanced users) */}
                <Accordion>
                  <Accordion.Item eventKey="raw">
                    <Accordion.Header>
                      🔬 Advanced: Raw CSV Data (for export/analysis)
                    </Accordion.Header>
                    <Accordion.Body>
                      {csvFiles.map(file => {
                        const data = csvData[file];
                        if (!data || data.length === 0) {
                          return (
                            <div key={file} className="mb-4">
                              <h6 className="text-muted">{file}</h6>
                              <Spinner animation="border" size="sm" />
                            </div>
                          );
                        }
                        const columns = Object.keys(data[0]);
                        const downloadUrl = `/api/simulations/${simulationId}/results/${file}?download=true`;

                        return (
                          <div key={file} className="mb-4">
                            <div className="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center gap-2 mb-2">
                              <h6 className="text-muted font-monospace small mb-0">{file}</h6>
                              <div className="d-flex flex-column flex-sm-row gap-2 w-100 w-md-auto">
                                <a
                                  href={downloadUrl}
                                  download={file.split('/').pop()}
                                  className="btn btn-outline-primary btn-sm d-flex align-items-center justify-content-center"
                                  title="Download CSV file to your computer"
                                >
                                  <span className="material-symbols-outlined">download</span>
                                </a>
                                <OutlineButton
                                  onClick={() => copyCSVToClipboard(data, file)}
                                  title="Copy data to clipboard for pasting into Excel/Google Sheets"
                                  className="d-flex align-items-center justify-content-center"
                                >
                                  <span className="material-symbols-outlined">content_copy</span>
                                </OutlineButton>
                              </div>
                            </div>
                            <div style={{ overflowX: 'auto', fontSize: '0.75rem' }}>
                              <Table striped bordered hover size="sm">
                                <thead>
                                  <tr>
                                    {columns.map(col => (
                                      <th key={col} className="font-monospace">
                                        {col}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {data.map((row, idx) => (
                                    <tr key={idx}>
                                      {columns.map(col => (
                                        <td key={col} className="font-monospace">
                                          {row[col]}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </div>
                          </div>
                        );
                      })}
                    </Accordion.Body>
                  </Accordion.Item>
                </Accordion>
              </>
            ) : (
              <Alert variant="info">
                No metrics available yet. Metrics will appear once the simulation completes.
              </Alert>
            )}
          </div>
        </Tab>

        <Tab eventKey="config" title="Config">
          <Card className="mt-3">
            <Card.Body>
              <ConfigDisplay config={config} />
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </div>
  );
}

export default SimulationDetails;
