import { useState } from 'react';
import { Card, Badge, ProgressBar, Button } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';
import { StrategyConfigModal } from './StrategyConfigModal';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';
import { RunningStrategyModal } from './RunningStrategyModal';
import { FailedStrategyModal } from './FailedStrategyModal';

export function QueueJobCard({ strategy, simulationId, sharedConfig, onConfigUpdate }) {
  const { index, config, status } = strategy;
  const navigate = useNavigate();

  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showRunningModal, setShowRunningModal] = useState(false);
  const [showFailedModal, setShowFailedModal] = useState(false);

  const getStatusBadge = () => {
    switch (status) {
      case 'running':
        return <Badge bg="primary">Running</Badge>;
      case 'completed':
        return <Badge bg="success">Completed</Badge>;
      case 'failed':
        return <Badge bg="danger">Failed</Badge>;
      case 'queued':
      default:
        return <Badge bg="secondary">Queued</Badge>;
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <MaterialIcon name="progress_activity" className="queue-job-spin" size={20} />;
      case 'completed':
        return <MaterialIcon name="check_circle" className="text-success" size={20} />;
      case 'failed':
        return <MaterialIcon name="cancel" className="text-danger" size={20} />;
      case 'queued':
      default:
        return <MaterialIcon name="schedule" className="text-muted" size={20} />;
    }
  };

  const handleCardClick = e => {
    if (e.target.closest('button, a')) {
      return;
    }

    switch (status) {
      case 'queued':
        setShowConfigModal(true);
        break;
      case 'running':
        setShowRunningModal(true);
        break;
      case 'completed':
        navigate(`/simulations/${simulationId}`);
        break;
      case 'failed':
        setShowFailedModal(true);
        break;
      default:
        break;
    }
  };

  const handleSaveConfig = (strategyIndex, newConfig) => {
    if (onConfigUpdate) {
      onConfigUpdate(strategyIndex, newConfig);
    }
  };

  const isClickable = ['queued', 'running', 'completed', 'failed'].includes(status);

  return (
    <>
      <Card
        className={`mb-3 queue-job-card ${isClickable ? 'queue-job-card-clickable' : ''}`}
        onClick={handleCardClick}
        role={isClickable ? 'button' : undefined}
        tabIndex={isClickable ? 0 : undefined}
        onKeyDown={
          isClickable
            ? e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleCardClick(e);
                }
              }
            : undefined
        }
      >
        <Card.Body>
          <div className="d-flex justify-content-between align-items-start mb-2">
            <div className="flex-grow-1">
              <h6 className="mb-1">
                {getStatusIcon()}
                <span className="ms-2">Strategy {index + 1}</span>
              </h6>
              <div className="text-muted small">
                <strong>{config.aggregation_strategy_keyword || 'fedavg'}</strong>
                {config.num_krum_selections && ` • k=${config.num_krum_selections}`}
                {config.trim_ratio && ` • trim=${config.trim_ratio}`}
              </div>
              <div className="text-muted small">
                {config.num_of_malicious_clients || 0} malicious clients
                {config.remove_clients === 'true' && ' • remove=true'}
              </div>
            </div>
            <div className="d-flex flex-column align-items-end gap-1">{getStatusBadge()}</div>
          </div>

          {status === 'running' && (
            <div className="mt-3">
              <ProgressBar animated now={100} variant="primary" className="queue-job-progress" />
              <div className="text-muted small mt-1">Executing...</div>
            </div>
          )}

          {status === 'completed' && (
            <div className="mt-3">
              <Button
                as={Link}
                to={`/simulations/${simulationId}`}
                variant="outline-primary"
                size="sm"
              >
                <MaterialIcon name="trending_up" size={20} className="me-2" />
                View Results
              </Button>
            </div>
          )}

          {status === 'failed' && (
            <div className="mt-2">
              <div className="alert alert-danger small mb-0">
                <MaterialIcon name="warning" size={20} className="me-2" />
                Strategy execution failed
              </div>
            </div>
          )}
        </Card.Body>
      </Card>

      <StrategyConfigModal
        show={showConfigModal}
        onHide={() => setShowConfigModal(false)}
        strategy={strategy}
        onSave={handleSaveConfig}
      />

      <RunningStrategyModal
        show={showRunningModal}
        onHide={() => setShowRunningModal(false)}
        strategy={strategy}
        simulationId={simulationId}
        sharedConfig={sharedConfig}
      />

      <FailedStrategyModal
        show={showFailedModal}
        onHide={() => setShowFailedModal(false)}
        strategy={strategy}
      />
    </>
  );
}
