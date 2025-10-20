import { Modal, Button, Alert } from 'react-bootstrap';

export function FailedStrategyModal({ show, onHide, strategy }) {
  if (!strategy) return null;

  const strategyName = strategy.config?.aggregation_strategy_keyword || 'fedavg';
  const displayName = strategyName.charAt(0).toUpperCase() + strategyName.slice(1);

  return (
    <Modal show={show} onHide={onHide} centered>
      <Modal.Header closeButton>
        <Modal.Title>
          <i className="bi bi-x-circle text-danger me-2"></i>
          Strategy {strategy.index + 1} Failed
        </Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Alert variant="danger" className="mb-3">
          <div className="d-flex align-items-start">
            <i className="bi bi-exclamation-triangle me-2 mt-1"></i>
            <div>
              <strong>Execution Failed</strong>
              <p className="mb-0 small mt-1">
                This strategy encountered an error during execution and could not complete.
              </p>
            </div>
          </div>
        </Alert>

        <div className="mb-3">
          <h6 className="mb-2">Strategy Configuration</h6>
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

        <div className="bg-light p-3 rounded">
          <h6 className="mb-2 small">Full Configuration</h6>
          <pre className="mb-0 small" style={{ maxHeight: '200px', overflow: 'auto' }}>
            <code>{JSON.stringify(strategy.config, null, 2)}</code>
          </pre>
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
