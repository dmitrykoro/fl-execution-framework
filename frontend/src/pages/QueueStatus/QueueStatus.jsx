import { useParams } from 'react-router-dom';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { Alert, Spinner, ProgressBar, Button } from 'react-bootstrap';
import { QueueJobCard } from '@components/features/experiment-queue/QueueJobCard';
import { useQueueStatus } from '@hooks/useQueueStatus';
import { stopSimulation } from '@api';
import { toast } from 'sonner';
import { useState } from 'react';

export function QueueStatus() {
  const { simulationId } = useParams();
  const { simulation, status, progress, loading, error } = useQueueStatus(simulationId);
  const [stopping, setStopping] = useState(false);

  const handleStopQueue = async () => {
    if (!confirm('Are you sure you want to stop this experiment queue?')) {
      return;
    }

    setStopping(true);
    try {
      await stopSimulation(simulationId);
      toast.success('Queue stopped successfully');
    } catch (err) {
      console.error('Failed to stop queue:', err);
      toast.error('Failed to stop queue');
    } finally {
      setStopping(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <div className="text-center py-5">
          <Spinner animation="border" variant="primary" />
          <p className="mt-3 text-muted">Loading queue status...</p>
        </div>
      </PageContainer>
    );
  }

  if (error) {
    return (
      <PageContainer>
        <Alert variant="danger">
          <strong>Error:</strong> Failed to load queue status
        </Alert>
      </PageContainer>
    );
  }

  // Check if this is a multi-simulation
  const isMultiSim = simulation?.config?.simulation_strategies?.length > 0;

  if (!isMultiSim) {
    return (
      <PageContainer>
        <Alert variant="warning">
          <i className="bi bi-exclamation-triangle me-2"></i>
          This simulation is not an experiment queue. It's a single simulation.
        </Alert>
      </PageContainer>
    );
  }

  const progressPercent = progress ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <PageContainer>
      <PageHeader title="Experiment Queue Status">
        <div className="d-flex gap-2">
          {status?.status === 'running' && (
            <Button
              variant="outline-danger"
              size="sm"
              onClick={handleStopQueue}
              disabled={stopping}
            >
              {stopping ? (
                <>
                  <Spinner as="span" animation="border" size="sm" className="me-2" />
                  Stopping...
                </>
              ) : (
                <>
                  <i className="bi bi-stop-circle me-2"></i>
                  Stop Queue
                </>
              )}
            </Button>
          )}
        </div>
      </PageHeader>

      {/* Progress Overview */}
      <Alert variant={progress?.isComplete ? 'success' : 'info'} className="mb-4">
        <div className="d-flex justify-content-between align-items-center mb-2">
          <div>
            <strong>
              {progress?.isComplete
                ? 'Queue Complete'
                : `Strategy ${(progress?.current || 0) + 1} of ${progress?.total || 0}`}
            </strong>
          </div>
          <div>
            <strong>{progressPercent}%</strong>
          </div>
        </div>
        <ProgressBar
          now={progressPercent}
          variant={progress?.isComplete ? 'success' : 'primary'}
          striped={!progress?.isComplete}
          animated={status?.status === 'running'}
        />
        <div className="mt-2 small text-muted">
          {progress?.isComplete
            ? 'All strategies have completed execution'
            : status?.status === 'running'
              ? 'Queue is currently executing...'
              : 'Queue execution paused or failed'}
        </div>
      </Alert>

      {/* Strategy Cards */}
      <h5 className="mb-3">Strategy Execution Progress</h5>
      {progress?.strategies.map(strategy => (
        <QueueJobCard key={strategy.index} strategy={strategy} simulationId={simulationId} />
      ))}

      {progress?.isComplete && (
        <div className="mt-4">
          <Alert variant="success">
            <i className="bi bi-check-circle me-2"></i>
            <strong>All done!</strong> All strategies in this experiment queue have completed. You
            can now compare results across all strategies.
          </Alert>
        </div>
      )}
    </PageContainer>
  );
}
