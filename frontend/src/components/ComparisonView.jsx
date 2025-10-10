import { useState, useEffect, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, Row, Col, Spinner, Alert, Table, Badge } from 'react-bootstrap';
import { getSimulationDetails, getSimulationStatus } from '../api';

function ComparisonView() {
  const [searchParams] = useSearchParams();
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

    const getSettings = config => config.shared_settings || config;

    const baseConfig = getSettings(simulations[0].details.config);
    const diffs = [];

    Object.keys(baseConfig).forEach(key => {
      const values = simulations.map(sim => getSettings(sim.details.config)[key]);
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

        const settings = sim.details.config.shared_settings || sim.details.config;

        return {
          id: sim.id,
          strategyName: settings.aggregation_strategy_keyword,
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
          rounds: settings.num_of_rounds,
          clients: settings.num_of_clients,
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
      </div>
    );
  }

  const configDiffs = getConfigDiff();
  const metricsComparison = getMetricsComparison();

  return (
    <div>
      <h1 className="mb-4">📊 Simulation Comparison</h1>

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
              {simulations.map(sim => {
                const settings = sim.details.config.shared_settings || sim.details.config;
                const displayName = settings.display_name;
                return (
                  <tr key={sim.id}>
                    <td>
                      <div>
                        {displayName && <div className="fw-semibold mb-1">{displayName}</div>}
                        <code className="small">{sim.id}</code>
                      </div>
                    </td>
                    <td>{settings.aggregation_strategy_keyword}</td>
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
                    <td>{settings.num_of_rounds}</td>
                    <td>{settings.num_of_clients}</td>
                  </tr>
                );
              })}
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
            {simulations.map((sim, idx) => {
              const plotFiles = sim.details.result_files.filter(
                file => file.endsWith('.pdf') && !file.includes('/')
              );

              const settings = sim.details.config.shared_settings || sim.details.config;
              const displayName = settings.display_name;

              return (
                <Col key={idx}>
                  <Card>
                    <Card.Header>
                      <div>
                        <small className="text-muted">Simulation {idx + 1}</small>
                        <div className="mt-1">
                          {displayName && <div className="fw-semibold">{displayName}</div>}
                          <code className="small">{sim.id}</code>
                        </div>
                      </div>
                    </Card.Header>
                    <Card.Body>
                      {plotFiles.length > 0 ? (
                        <div className="d-flex flex-column gap-2">
                          {plotFiles.map((filename, plotIdx) => (
                            <div key={plotIdx}>
                              <a
                                href={`http://localhost:8000/api/simulations/${sim.id}/results/${filename}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-decoration-none"
                              >
                                📈 {filename.replace('.pdf', '').replace(/_/g, ' ')}
                              </a>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-muted">No plots available</p>
                      )}
                    </Card.Body>
                  </Card>
                </Col>
              );
            })}
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
}

export default ComparisonView;
