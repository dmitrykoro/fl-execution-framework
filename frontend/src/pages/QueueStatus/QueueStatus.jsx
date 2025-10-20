import { useParams } from 'react-router-dom';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { Alert, Spinner, ProgressBar, Button } from 'react-bootstrap';
import { QueueJobCard } from '@components/features/experiment-queue/QueueJobCard';
import { ConfirmModal } from '@components/common/Modal/ConfirmModal';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';
import { useQueueStatus } from '@hooks/useQueueStatus';
import { stopSimulation } from '@api';
import { toast } from 'sonner';
import { useState } from 'react';

export function QueueStatus() {
  const { simulationId } = useParams();
  const { simulation, status, progress, loading, error } = useQueueStatus(simulationId);
  const [stopping, setStopping] = useState(false);
  const [showStopModal, setShowStopModal] = useState(false);

  const handleStopQueue = () => {
    setShowStopModal(true);
  };

  const confirmStopQueue = async () => {
    setStopping(true);
    setShowStopModal(false);
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

  const isMultiSim = simulation?.config?.simulation_strategies?.length > 0;

  if (!isMultiSim) {
    return (
      <PageContainer>
        <Alert variant="warning">
          <MaterialIcon name="warning" size={20} className="me-2" />
          This simulation is not an experiment queue. It's a single simulation.
        </Alert>
      </PageContainer>
    );
  }

  const progressPercent = progress ? Math.round((progress.current / progress.total) * 100) : 0;

  const strategyGroups = progress?.strategies
    ? progress.strategies.reduce((groups, strategy) => {
        const key = strategy.config.aggregation_strategy_keyword || 'fedavg';
        if (!groups[key]) {
          groups[key] = {
            name: key,
            strategies: [],
            completed: 0,
            total: 0,
          };
        }
        groups[key].strategies.push(strategy);
        groups[key].total++;
        if (strategy.status === 'completed') {
          groups[key].completed++;
        }
        return groups;
      }, {})
    : null;

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
                  <MaterialIcon name="stop_circle" size={20} className="me-2" />
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

      {/* Strategy Groups Overview */}
      {strategyGroups && Object.keys(strategyGroups).length > 1 && (
        <div className="mb-4">
          <h6 className="mb-3">📊 Strategy Groups</h6>
          <div className="d-flex flex-wrap gap-2">
            {Object.values(strategyGroups).map(group => (
              <Alert key={group.name} variant="light" className="mb-0 py-2 px-3">
                <strong>{group.name}</strong>:{' '}
                <span className={group.completed === group.total ? 'text-success' : 'text-primary'}>
                  {group.completed}/{group.total} complete
                </span>
              </Alert>
            ))}
          </div>
        </div>
      )}

      {/* Strategy Cards */}
      <h5 className="mb-3">Strategy Execution Progress</h5>
      {progress?.strategies.map(strategy => (
        <QueueJobCard
          key={strategy.index}
          strategy={strategy}
          simulationId={simulationId}
          sharedConfig={simulation?.config?.shared_settings}
        />
      ))}

      {progress?.isComplete && (
        <div className="mt-4">
          <Alert variant="success">
            <MaterialIcon name="check_circle" size={20} className="me-2" />
            <strong>All done!</strong> All strategies in this experiment queue have completed. You
            can now compare results across all strategies.
          </Alert>
        </div>
      )}

      <ConfirmModal
        show={showStopModal}
        title="Stop Queue"
        message="Are you sure you want to stop this experiment queue?"
        variant="danger"
        onConfirm={confirmStopQueue}
        onCancel={() => setShowStopModal(false)}
      />
    </PageContainer>
  );
}
