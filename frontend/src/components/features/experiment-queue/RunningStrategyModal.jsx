import { Modal, Button, ProgressBar, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

export function RunningStrategyModal({ show, onHide, strategy, simulationId, sharedConfig }) {
  const navigate = useNavigate();

  const handleViewDetails = () => {
    onHide();
    navigate(`/simulations/${simulationId}`);
  };

  if (!strategy) return null;

  const strategyName = strategy.config?.aggregation_strategy_keyword || 'fedavg';
  const displayName = strategyName.charAt(0).toUpperCase() + strategyName.slice(1);

  return (
    <Modal show={show} onHide={onHide} centered>
      <Modal.Header closeButton>
        <Modal.Title>
          <i className="bi bi-arrow-repeat queue-job-spin me-2"></i>
          {displayName} Defense
        </Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <div className="mb-3">
          <div className="d-flex align-items-center gap-2 mb-3">
            <Spinner animation="border" size="sm" variant="primary" />
            <h6 className="mb-0">Running</h6>
          </div>

          <div className="mb-3">
            <small className="text-muted d-block mb-1">Simulation ID</small>
            <code className="bg-light px-2 py-1 rounded">{simulationId}</code>
          </div>

          <ProgressBar animated now={100} variant="primary" className="mb-3" />

          <div className="bg-light p-3 rounded">
            <h6 className="mb-2">Simulation in Progress</h6>
            <p className="mb-1 small">
              Running federated learning simulation with{' '}
              <strong>{sharedConfig?.num_of_clients || 'N/A'} clients</strong> over{' '}
              <strong>{sharedConfig?.num_of_rounds || 'N/A'} rounds</strong>...
            </p>
            <p className="mb-0 small text-muted">
              Status updates every 2 seconds. Results will appear automatically when complete.
            </p>
          </div>
        </div>

        <div className="mb-2">
          <small className="text-muted d-block mb-2">Strategy Configuration</small>
          <div className="small">
            <div className="d-flex justify-content-between py-1">
              <span className="text-muted">Aggregation:</span>
              <strong>{displayName}</strong>
            </div>
            <div className="d-flex justify-content-between py-1">
              <span className="text-muted">Malicious clients:</span>
              <strong>{strategy.config?.num_of_malicious_clients || 0}</strong>
            </div>
            {strategy.config?.num_krum_selections && (
              <div className="d-flex justify-content-between py-1">
                <span className="text-muted">Krum selections (k):</span>
                <strong>{strategy.config.num_krum_selections}</strong>
              </div>
            )}
            {strategy.config?.remove_clients !== undefined && (
              <div className="d-flex justify-content-between py-1">
                <span className="text-muted">Clients to remove:</span>
                <strong>{strategy.config.remove_clients}</strong>
              </div>
            )}
          </div>
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
        <Button variant="primary" onClick={handleViewDetails}>
          <i className="bi bi-graph-up me-2"></i>
          View Simulation Details
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
