import { useState, useEffect } from 'react';
import { Alert, Spinner, Card, Table, Badge } from 'react-bootstrap';
import { getAllPlotData } from '@api/endpoints/simulations';
import { StrategyComparisonPlot } from './StrategyComparisonPlot';
import { ConfigExplainer } from '@components/features/education/ConfigExplainer';

export function ComparisonTab({ simulation, isMultiStrategy }) {
  const [allPlotData, setAllPlotData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState(null);

  useEffect(() => {
    if (!isMultiStrategy || simulation.status !== 'completed') {
      setLoading(false);
      return;
    }

    const fetchAllPlotData = async () => {
      try {
        const response = await getAllPlotData(simulation.id);
        setAllPlotData(response.data.strategies);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch comparison data:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchAllPlotData();
  }, [simulation.id, simulation.status, isMultiStrategy]);

  if (!isMultiStrategy) {
    return (
      <Card className="mt-3">
        <Card.Body>
          <Alert variant="info">
            Strategy comparison is only available for multi-strategy experiments.
          </Alert>
        </Card.Body>
      </Card>
    );
  }

  if (simulation.status !== 'completed') {
    return (
      <Card className="mt-3">
        <Card.Body>
          <Alert variant="info">
            ⏳ Comparison view will be available when all strategies complete...
          </Alert>
        </Card.Body>
      </Card>
    );
  }

  if (loading) {
    return (
      <div className="text-center p-4">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading comparison data...</span>
        </Spinner>
        <p className="mt-2">Loading comparison data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="mt-3">
        <Card.Body>
          <Alert variant="danger">Failed to load comparison data: {error}</Alert>
        </Card.Body>
      </Card>
    );
  }

  if (!allPlotData || allPlotData.length === 0) {
    return (
      <Card className="mt-3">
        <Card.Body>
          <Alert variant="warning">No comparison data available yet</Alert>
        </Card.Body>
      </Card>
    );
  }

  const strategyConfigs = {};
  simulation.config.simulation_strategies.forEach((config, index) => {
    strategyConfigs[index] = {
      ...simulation.config.shared_settings,
      ...config,
    };
  });

  const performanceSummary = allPlotData.map(strategy => {
    const config = strategyConfigs[strategy.strategy_number];
    const roundMetrics = strategy.data.round_metrics;

    const finalAccuracy = roundMetrics?.average_accuracy_history
      ? roundMetrics.average_accuracy_history[roundMetrics.average_accuracy_history.length - 1]
      : null;

    const finalLoss = roundMetrics?.average_loss_history
      ? roundMetrics.average_loss_history[roundMetrics.average_loss_history.length - 1]
      : null;

    const removalAccuracy = roundMetrics?.removal_accuracy_history
      ? roundMetrics.removal_accuracy_history[roundMetrics.removal_accuracy_history.length - 1]
      : null;

    return {
      strategyNumber: strategy.strategy_number,
      aggregationStrategy: config.aggregation_strategy_keyword || 'fedavg',
      numKrumSelections: config.num_krum_selections,
      numMalicious: config.num_of_malicious_clients || 0,
      removeClients: config.remove_clients === 'true',
      finalAccuracy,
      finalLoss,
      removalAccuracy,
    };
  });

  const sortedByAccuracy = [...performanceSummary].sort(
    (a, b) => (b.finalAccuracy || 0) - (a.finalAccuracy || 0)
  );
  const bestPerformer = sortedByAccuracy[0];
  const worstPerformer = sortedByAccuracy[sortedByAccuracy.length - 1];

  return (
    <div className="mt-3">
      <h5 className="mb-3">🔬 Multi-Strategy Comparison</h5>
      <p className="text-muted mb-4">
        Comparing {allPlotData.length} strategy variations across different configurations and
        attack scenarios.
      </p>

      {/* Performance Summary Table */}
      <Card className="mb-4">
        <Card.Body>
          <h6 className="mb-3">📊 Performance Summary</h6>
          <div className="table-responsive">
            <Table striped bordered hover size="sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Strategy</th>
                  <th>Config</th>
                  <th>Final Accuracy</th>
                  <th>Final Loss</th>
                  <th>Defense Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {sortedByAccuracy.map(summary => (
                  <tr
                    key={summary.strategyNumber}
                    onClick={() => setSelectedStrategy(summary.strategyNumber)}
                    style={{ cursor: 'pointer' }}
                    className={selectedStrategy === summary.strategyNumber ? 'table-active' : ''}
                  >
                    <td>
                      <Badge bg="secondary">{summary.strategyNumber}</Badge>
                      {summary.strategyNumber === bestPerformer.strategyNumber && (
                        <Badge bg="success" className="ms-1">
                          🏆 Best
                        </Badge>
                      )}
                    </td>
                    <td>
                      <strong>{summary.aggregationStrategy}</strong>
                      {summary.numKrumSelections && ` (k=${summary.numKrumSelections})`}
                    </td>
                    <td>
                      <small>
                        {summary.numMalicious} malicious
                        {summary.removeClients && ' • rm=true'}
                      </small>
                    </td>
                    <td>
                      <Badge
                        bg={
                          summary.finalAccuracy >= 0.9
                            ? 'success'
                            : summary.finalAccuracy >= 0.7
                              ? 'warning'
                              : 'danger'
                        }
                      >
                        {summary.finalAccuracy !== null
                          ? `${(summary.finalAccuracy * 100).toFixed(1)}%`
                          : 'N/A'}
                      </Badge>
                    </td>
                    <td>{summary.finalLoss !== null ? summary.finalLoss.toFixed(4) : 'N/A'}</td>
                    <td>
                      {summary.removalAccuracy !== null
                        ? `${(summary.removalAccuracy * 100).toFixed(0)}%`
                        : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
          <small className="text-muted">💡 Click a row to view detailed strategy explanation</small>
        </Card.Body>
      </Card>

      {/* Strategy Explanation (when selected) */}
      {selectedStrategy !== null && (
        <ConfigExplainer
          strategy={strategyConfigs[selectedStrategy].aggregation_strategy_keyword}
          config={strategyConfigs[selectedStrategy]}
        />
      )}

      {/* Key Insights */}
      <Card className="mb-4">
        <Card.Body>
          <h6 className="mb-3">🎯 Key Insights</h6>
          <div className="d-flex flex-column gap-2">
            {bestPerformer && (
              <Alert variant="success" className="mb-2">
                <strong>🏆 Best Performer:</strong> Strategy {bestPerformer.strategyNumber} (
                {bestPerformer.aggregationStrategy}
                {bestPerformer.numKrumSelections && ` k=${bestPerformer.numKrumSelections}`}) with{' '}
                {bestPerformer.numMalicious} malicious clients achieved{' '}
                {(bestPerformer.finalAccuracy * 100).toFixed(1)}% accuracy
              </Alert>
            )}

            {worstPerformer && bestPerformer !== worstPerformer && (
              <Alert variant="warning" className="mb-2">
                <strong>⚠️ Most Vulnerable:</strong> Strategy {worstPerformer.strategyNumber} (
                {worstPerformer.aggregationStrategy}) with {worstPerformer.numMalicious} malicious
                clients had lowest accuracy at {(worstPerformer.finalAccuracy * 100).toFixed(1)}%
              </Alert>
            )}

            <Alert variant="info" className="mb-0">
              <strong>📈 Experiment Scope:</strong> This comparison includes {allPlotData.length}{' '}
              strategy variations testing{' '}
              {new Set(performanceSummary.map(s => s.aggregationStrategy)).size} different
              aggregation strategies across various attack scenarios
            </Alert>
          </div>
        </Card.Body>
      </Card>

      {/* Interactive Comparison Plot */}
      <StrategyComparisonPlot allPlotData={allPlotData} strategyConfigs={strategyConfigs} />

      {/* Educational Note */}
      <Card className="mb-4">
        <Card.Body>
          <h6 className="mb-2">🎓 Understanding This Comparison</h6>
          <p className="text-muted mb-0">
            This view allows you to compare how different aggregation strategies and configurations
            perform under varying attack conditions. Use the interactive plot above to toggle
            strategies and explore their behavior across training rounds. Click on individual
            strategies in the performance table to learn more about each approach.
          </p>
        </Card.Body>
      </Card>
    </div>
  );
}
