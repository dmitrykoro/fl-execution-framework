import React, { useState, useEffect, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Spinner, Alert, Button, Table, Badge } from 'react-bootstrap';
import { getSimulationDetails, getSimulationStatus } from '../api';

function ComparisonView() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const simulationIds = useMemo(() => searchParams.get('ids')?.split(',') || [], [searchParams]);

  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (simulationIds.length < 2) {
      setError('Please select at least 2 simulations to compare');
      setLoading(false);
      return;
    }

    const fetchSimulations = async () => {
      try {
        const promises = simulationIds.map(async id => {
          const [detailsRes, statusRes] = await Promise.all([
            getSimulationDetails(id),
            getSimulationStatus(id),
          ]);
          return {
            id,
            details: detailsRes.data,
            status: statusRes.data,
          };
        });

        const results = await Promise.all(promises);
        setSimulations(results);
        setLoading(false);
      } catch (err) {
        setError(`Failed to load simulations: ${err.message}`);
        setLoading(false);
      }
    };

    fetchSimulations();
  }, [simulationIds]);

  const getConfigDiff = () => {
    if (simulations.length < 2) return [];

    const baseConfig = simulations[0].details.config;
    const diffs = [];

    Object.keys(baseConfig).forEach(key => {
      const values = simulations.map(sim => sim.details.config[key]);
      const allSame = values.every(v => JSON.stringify(v) === JSON.stringify(values[0]));

      if (!allSame) {
        diffs.push({ key, values });
      }
    });

    return diffs;
  };

  const getMetricsComparison = () => {
    return simulations
      .map(sim => {
        const csvData = sim.details.csv_data || {};
        const roundMetrics = csvData['csv/round_metrics_0.csv'] || [];

        if (roundMetrics.length === 0) return null;

        const firstRound = roundMetrics[0];
        const lastRound = roundMetrics[roundMetrics.length - 1];

        return {
          id: sim.id,
          strategyName: sim.details.config.strategy_name,
          initialAccuracy: firstRound?.test_accuracy || 0,
          finalAccuracy: lastRound?.test_accuracy || 0,
          improvement:
            lastRound?.test_accuracy && firstRound?.test_accuracy
              ? (
                  ((lastRound.test_accuracy - firstRound.test_accuracy) /
                    firstRound.test_accuracy) *
                  100
                ).toFixed(1)
              : 'N/A',
          rounds: sim.details.config.num_of_rounds,
          clients: sim.details.config.num_of_clients,
        };
      })
      .filter(m => m !== null);
  };

  if (loading) {
    return (
      <div>
        <h1>📊 Simulation Comparison</h1>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>📊 Simulation Comparison</h1>
        <Alert variant="danger">{error}</Alert>
        <Button onClick={() => navigate('/')}>Back to Dashboard</Button>
      </div>
    );
  }

  const configDiffs = getConfigDiff();
  const metricsComparison = getMetricsComparison();

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1>📊 Simulation Comparison</h1>
        <Button variant="secondary" onClick={() => navigate('/')}>
          Back to Dashboard
        </Button>
      </div>

      <Card className="mb-4">
        <Card.Header>
          <h5>🎯 Overview</h5>
        </Card.Header>
        <Card.Body>
          <Table responsive>
            <thead>
              <tr>
                <th>Simulation ID</th>
                <th>Strategy</th>
                <th>Status</th>
                <th>Rounds</th>
                <th>Clients</th>
              </tr>
            </thead>
            <tbody>
              {simulations.map(sim => (
                <tr key={sim.id}>
                  <td>
                    <code>{sim.id}</code>
                  </td>
                  <td>{sim.details.config.strategy_name}</td>
                  <td>
                    <Badge
                      bg={
                        sim.status.status === 'completed'
                          ? 'success'
                          : sim.status.status === 'running'
                            ? 'primary'
                            : sim.status.status === 'failed'
                              ? 'danger'
                              : 'secondary'
                      }
                    >
                      {sim.status.status}
                    </Badge>
                  </td>
                  <td>{sim.details.config.num_of_rounds}</td>
                  <td>{sim.details.config.num_of_clients}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card.Body>
      </Card>

      {metricsComparison.length > 0 && (
        <Card className="mb-4">
          <Card.Header>
            <h5>📈 Performance Metrics</h5>
          </Card.Header>
          <Card.Body>
            <Table responsive>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Initial Accuracy</th>
                  <th>Final Accuracy</th>
                  <th>Improvement</th>
                </tr>
              </thead>
              <tbody>
                {metricsComparison.map(metric => (
                  <tr key={metric.id}>
                    <td>{metric.strategyName}</td>
                    <td>{(metric.initialAccuracy * 100).toFixed(2)}%</td>
                    <td>{(metric.finalAccuracy * 100).toFixed(2)}%</td>
                    <td>
                      <Badge bg={parseFloat(metric.improvement) > 0 ? 'success' : 'warning'}>
                        {metric.improvement}%
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card.Body>
        </Card>
      )}

      {configDiffs.length > 0 ? (
        <Card className="mb-4">
          <Card.Header>
            <h5>⚙️ Configuration Differences</h5>
          </Card.Header>
          <Card.Body>
            <Table responsive>
              <thead>
                <tr>
                  <th>Parameter</th>
                  {simulations.map((sim, idx) => (
                    <th key={idx}>Sim {idx + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {configDiffs.map(diff => (
                  <tr key={diff.key}>
                    <td>
                      <strong>{diff.key}</strong>
                    </td>
                    {diff.values.map((value, idx) => (
                      <td key={idx}>
                        <code>{JSON.stringify(value)}</code>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card.Body>
        </Card>
      ) : (
        <Alert variant="info">All configurations are identical across selected simulations.</Alert>
      )}

      <Card className="mb-4">
        <Card.Header>
          <h5>📊 Side-by-Side Plots</h5>
        </Card.Header>
        <Card.Body>
          <Row xs={1} md={2} className="g-3">
            {simulations.map((sim, idx) => (
              <Col key={idx}>
                <Card>
                  <Card.Header>
                    <small>
                      Simulation {idx + 1}: {sim.id}
                    </small>
                  </Card.Header>
                  <Card.Body>
                    {sim.details.plots && sim.details.plots.length > 0 ? (
                      sim.details.plots.map((plot, plotIdx) => (
                        <div key={plotIdx} className="mb-3">
                          <img
                            src={`data:image/png;base64,${plot.data}`}
                            alt={plot.filename}
                            style={{ width: '100%', height: 'auto' }}
                          />
                          <small className="text-muted">{plot.filename}</small>
                        </div>
                      ))
                    ) : (
                      <p className="text-muted">No plots available</p>
                    )}
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
}

export default ComparisonView;
