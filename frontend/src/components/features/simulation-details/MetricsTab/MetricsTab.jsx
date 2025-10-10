import { Alert, Card, Accordion } from 'react-bootstrap';
import { RoundMetricsTable } from './RoundMetricsTable';
import { ExecutionStats } from './ExecutionStats';
import { RawDataAccordion } from './RawDataAccordion';

export function MetricsTab({ csvData, csvFiles, config, simulationId }) {
  if (csvFiles.length === 0) {
    return (
      <Alert variant="info" className="mt-3">
        No metrics available yet. Metrics will appear once the simulation completes.
      </Alert>
    );
  }

  const cfg = config.shared_settings || config;

  return (
    <div className="mt-3">
      {csvData['csv/round_metrics_0.csv'] && (
        <RoundMetricsTable data={csvData['csv/round_metrics_0.csv']} config={cfg} />
      )}

      {csvData['csv/exec_stats_0.csv'] && (
        <ExecutionStats data={csvData['csv/exec_stats_0.csv']} config={cfg} />
      )}

      <RawDataAccordion csvFiles={csvFiles} csvData={csvData} simulationId={simulationId} />
    </div>
  );
}
