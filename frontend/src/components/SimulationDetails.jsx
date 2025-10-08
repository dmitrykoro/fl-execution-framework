import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  Spinner,
  Alert,
  Card,
  Tabs,
  Tab,
  Table,
  Button,
  ProgressBar,
  ListGroup,
  Accordion,
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap';
import useApi from '../hooks/useApi';
import { getSimulationDetails, getSimulationStatus, getResultFile, createSimulation } from '../api';
import InteractivePlots from './InteractivePlots';

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
      const response = await createSimulation(config);
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
            <Button variant="outline-secondary" size="sm" onClick={() => setShowRawJSON(false)}>
              View Human-Readable
            </Button>
          </div>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
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
          <Button variant="outline-secondary" size="sm" onClick={() => setShowRawJSON(true)}>
            View Raw JSON
          </Button>
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

  return (
    <div>
      <Link to="/">&larr; Back to Dashboard</Link>
      <div className="d-flex align-items-center gap-3 mt-3 mb-3">
        <h4 className="mb-0 text-muted">
          Simulation: <span className="text-dark">{simulationId}</span>
        </h4>
        <span className={`badge bg-${statusVariant}`}>{displayStatus}</span>
        <Button variant="outline-primary" onClick={handleRunAgain} disabled={isCloning}>
          {isCloning ? 'Starting...' : 'Run Again'}
        </Button>
      </div>

      {displayStatus === 'running' && (
        <Card className="mb-3">
          <Card.Body>
            <div className="d-flex align-items-center gap-3 mb-2">
              <h6 className="mb-0">Simulation in Progress</h6>
              <Spinner animation="border" size="sm" />
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

      <Tabs defaultActiveKey="insights" className="mt-4">
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
              csvFiles.map(file => {
                const data = csvData[file];
                if (!data || data.length === 0) {
                  return (
                    <div key={file} className="mb-4">
                      <h5>{file}</h5>
                      <Spinner animation="border" size="sm" />
                    </div>
                  );
                }
                const columns = Object.keys(data[0]);
                return (
                  <div key={file} className="mb-4">
                    <h5>{file}</h5>
                    <div style={{ overflowX: 'auto' }}>
                      <Table striped bordered hover size="sm">
                        <thead>
                          <tr>
                            {columns.map(col => (
                              <th key={col}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.map((row, idx) => (
                            <tr key={idx}>
                              {columns.map(col => (
                                <td key={col}>{row[col]}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-muted">No metrics available</p>
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
