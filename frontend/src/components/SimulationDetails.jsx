import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Spinner, Alert, Card, Row, Col, Tabs, Tab, Table, Button, ProgressBar, ListGroup } from 'react-bootstrap';
import useApi from '../hooks/useApi';
import { getSimulationDetails, getSimulationStatus, getResultFile, createSimulation } from '../api';

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
      csvFiles.forEach(async (file) => {
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
  const plotFiles = result_files.filter(file => file.endsWith('.png') || file.endsWith('.pdf'));
  const csvFiles = result_files.filter(file => file.endsWith('.csv'));
  const otherFiles = result_files.filter(file => !file.endsWith('.png') && !file.endsWith('.pdf') && !file.endsWith('.csv'));

  const statusVariant = displayStatus === 'completed' ? 'success' : displayStatus === 'failed' ? 'danger' : displayStatus === 'running' ? 'primary' : 'secondary';

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
      const lastAccuracy = parseFloat(roundMetrics[roundMetrics.length - 1].average_accuracy_history);
      const improvement = ((lastAccuracy - firstAccuracy) / firstAccuracy * 100).toFixed(1);

      if (improvement > 0) {
        insights.push({
          type: 'success',
          icon: '📈',
          text: `Model accuracy improved by ${improvement}% over ${roundMetrics.length} rounds (from ${(firstAccuracy * 100).toFixed(1)}% to ${(lastAccuracy * 100).toFixed(1)}%)`
        });
      } else if (improvement < 0) {
        insights.push({
          type: 'warning',
          icon: '⚠️',
          text: `Model accuracy decreased by ${Math.abs(improvement)}% - this may indicate attack or poor hyperparameters`
        });
      }
    }

    // Analyze malicious clients
    if (config.num_of_malicious_clients > 0) {
      insights.push({
        type: 'info',
        icon: '🎯',
        text: `Simulation includes ${config.num_of_malicious_clients} malicious client(s) using ${config.attack_type} attack`
      });

      // Check removal metrics
      if (config.remove_clients === 'true' && roundMetrics.length > 0) {
        const lastRound = roundMetrics[roundMetrics.length - 1];
        const removalAccuracy = parseFloat(lastRound.removal_accuracy_history);
        const removalPrecision = parseFloat(lastRound.removal_precision_history);
        const removalRecall = parseFloat(lastRound.removal_recall_history);

        if (removalAccuracy === 1.0) {
          insights.push({
            type: 'success',
            icon: '✓',
            text: `Defense strategy (${config.aggregation_strategy_keyword}) successfully identified all malicious clients with 100% accuracy`
          });
        } else if (removalAccuracy >= 0.7) {
          insights.push({
            type: 'success',
            icon: '✓',
            text: `Defense detected malicious clients with ${(removalAccuracy * 100).toFixed(0)}% accuracy (Precision: ${(removalPrecision * 100).toFixed(0)}%, Recall: ${(removalRecall * 100).toFixed(0)}%)`
          });
        } else if (removalAccuracy > 0) {
          insights.push({
            type: 'warning',
            icon: '⚠️',
            text: `Defense partially effective: ${(removalAccuracy * 100).toFixed(0)}% accuracy in detecting malicious clients`
          });
        }
      }
    } else {
      insights.push({
        type: 'info',
        icon: 'ℹ️',
        text: 'Baseline simulation with no malicious clients - observing natural federated learning behavior'
      });
    }

    // Analyze client participation (from per_client_metrics)
    if (perClientMetrics && perClientMetrics.length > 0) {
      const lastRound = perClientMetrics[perClientMetrics.length - 1];
      const participationKeys = Object.keys(lastRound).filter(k => k.includes('aggregation_participation_history'));
      const activeClients = participationKeys.filter(k => lastRound[k] === '1').length;
      const removedClients = config.num_of_clients - activeClients;

      if (removedClients > 0) {
        insights.push({
          type: 'info',
          icon: '🔒',
          text: `${removedClients} client(s) removed from aggregation by round ${roundMetrics.length} (${activeClients} active clients remaining)`
        });
      }
    }

    // Analyze defense strategy behavior
    if (config.aggregation_strategy_keyword === 'pid' && config.remove_clients === 'true') {
      const beginRemoving = config.begin_removing_from_round || 2;
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `PID-based removal strategy started evaluating clients from round ${beginRemoving} with ${config.pid_p || 0.1} proportional gain`
      });
    } else if (config.aggregation_strategy_keyword === 'krum') {
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `Krum aggregation selects the most trustworthy client update based on distance metrics`
      });
    } else if (config.aggregation_strategy_keyword === 'trimmed_mean') {
      insights.push({
        type: 'info',
        icon: '🛡️',
        text: `Trimmed mean removes extreme updates before aggregation for robustness`
      });
    }

    // Dataset and model info
    insights.push({
      type: 'info',
      icon: '📊',
      text: `Trained ${config.model_type || 'cnn'} model on ${config.dataset_keyword} dataset with ${config.num_of_clients} clients`
    });

    return insights;
  };

  const insights = displayStatus === 'completed' ? generateInsights() : [];

  return (
    <div>
      <Link to="/">&larr; Back to Dashboard</Link>
      <div className="d-flex align-items-center gap-3 mt-3 mb-3">
        <h4 className="mb-0 text-muted">Simulation: <span className="text-dark">{simulationId}</span></h4>
        <span className={`badge bg-${statusVariant}`}>{displayStatus}</span>
        <Button
          variant="outline-primary"
          onClick={handleRunAgain}
          disabled={isCloning}
        >
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
              <p className="mb-1">Running federated learning simulation with {config.num_of_clients} clients over {config.num_of_rounds} rounds...</p>
              <p className="mb-0">Status updates every 2 seconds. Results will appear automatically when complete.</p>
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
                    Automatic analysis of simulation results to help understand federated learning behavior and defense effectiveness.
                  </p>
                  <ListGroup>
                    {insights.map((insight, idx) => (
                      <ListGroup.Item
                        key={idx}
                        variant={insight.type === 'success' ? 'success' : insight.type === 'warning' ? 'warning' : 'light'}
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
                  Insights will be generated once the simulation completes and metrics are available.
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="plots" title="Plots">
          <Row xs={1} md={2} className="g-4 mt-2">
            {plotFiles.length > 0 ? (
              plotFiles.map(file => (
                <Col key={file}>
                  <Card>
                    {file.endsWith('.pdf') ? (
                      <iframe
                        src={`/api/simulations/${simulationId}/results/${file}`}
                        style={{ width: '100%', height: '400px', border: 'none' }}
                        title={file}
                      />
                    ) : (
                      <Card.Img variant="top" src={`/api/simulations/${simulationId}/results/${file}`} />
                    )}
                    <Card.Body>
                      <Card.Title>{file}</Card.Title>
                    </Card.Body>
                  </Card>
                </Col>
              ))
            ) : (
              <Col><p className="text-muted">No plots available</p></Col>
            )}
          </Row>
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
            {otherFiles.length > 0 && (
              <>
                <h5 className="mt-3">Other Files</h5>
                <ul>
                  {otherFiles.map(file => (
                    <li key={file}>
                      <a href={`/api/simulations/${simulationId}/results/${file}`} target="_blank" rel="noopener noreferrer">
                        {file}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </Tab>

        <Tab eventKey="config" title="Config">
          <Card className="mt-3">
            <Card.Body>
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'monospace', fontSize: '0.875rem' }}>
                {JSON.stringify(config, null, 2)}
              </pre>
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </div>
  );
}

export default SimulationDetails;