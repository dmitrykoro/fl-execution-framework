import { Alert } from 'react-bootstrap';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { QueueBuilder } from '@components/features/experiment-queue/QueueBuilder';
import { useRunningSimulation } from '@hooks/useRunningSimulation';

export function ExperimentQueue() {
  const { hasRunning } = useRunningSimulation();

  return (
    <PageContainer>
      <PageHeader title="Experiment Queue">
        <p className="text-muted mb-0">
          Build multi-strategy experiments with sequential execution
        </p>
      </PageHeader>

      {hasRunning && (
        <Alert variant="info" className="mb-4">
          <i className="bi bi-info-circle me-2"></i>
          <strong>Note:</strong> A simulation is currently running. This experiment queue will start
          automatically after the current simulation completes.
        </Alert>
      )}

      <QueueBuilder />
    </PageContainer>
  );
}
