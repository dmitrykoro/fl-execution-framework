import { Card, Badge, ProgressBar, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';

export function QueueJobCard({ strategy, simulationId }) {
  const { index, config, status } = strategy;

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
        return <i className="bi bi-arrow-repeat queue-job-spin"></i>;
      case 'completed':
        return <i className="bi bi-check-circle text-success"></i>;
      case 'failed':
        return <i className="bi bi-x-circle text-danger"></i>;
      case 'queued':
      default:
        return <i className="bi bi-clock text-muted"></i>;
    }
  };

  return (
    <Card className="mb-3 queue-job-card">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-start mb-2">
          <div>
            <h6 className="mb-1">
              {getStatusIcon()}
              <span className="ms-2">Strategy {index + 1}</span>
            </h6>
            <div className="text-muted small">
              <strong>{config.aggregation_strategy_keyword || 'fedavg'}</strong>
              {' • '}
              {config.num_of_malicious_clients || 0} malicious
              {config.num_krum_selections && ` • k=${config.num_krum_selections}`}
            </div>
          </div>
          {getStatusBadge()}
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
              <i className="bi bi-graph-up me-2"></i>
              View Results
            </Button>
          </div>
        )}

        {status === 'failed' && (
          <div className="mt-2">
            <div className="alert alert-danger small mb-0">
              <i className="bi bi-exclamation-triangle me-2"></i>
              Strategy execution failed
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}
