import { Alert, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { QueueBuilder } from '@components/features/experiment-queue/QueueBuilder';
import { useRunningSimulation } from '@hooks/useRunningSimulation';

export function ExperimentQueue() {
  const { hasRunning, runningSimIds } = useRunningSimulation();

  return (
    <PageContainer>
      <PageHeader title="Experiment Queue" />

      {hasRunning && (
        <Alert variant="warning" className="mb-4">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <i className="bi bi-exclamation-triangle me-2"></i>
              <strong>Simulation in progress</strong> - New simulations will queue automatically
            </div>
            <Button
              as={Link}
              to={`/queue/${runningSimIds[0]}`}
              className="btn-warning-action"
              size="sm"
            >
              View Queue Status
            </Button>
          </div>
        </Alert>
      )}

      <QueueBuilder />
    </PageContainer>
  );
}
