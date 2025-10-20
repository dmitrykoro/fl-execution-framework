import { Modal, Button } from 'react-bootstrap';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';

export function QueueChoiceModal({ show, onHide, onAddToQueue, onCreateSeparate }) {
  return (
    <Modal show={show} onHide={onHide} centered>
      <Modal.Header closeButton>
        <Modal.Title>Simulation Already Running</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p className="mb-3">
          An experiment queue is currently running. How would you like to proceed?
        </p>

        <div className="d-grid gap-3">
          <Button
            variant="outline-primary"
            size="lg"
            onClick={onAddToQueue}
            className="text-start d-flex align-items-start gap-3 p-3"
          >
            <MaterialIcon name="add" size={24} className="mt-1" />
            <div>
              <div className="fw-bold mb-1">Add to Experiment Queue</div>
              <div className="small text-muted">
                Add this simulation as another strategy variation to the running queue for
                comparison
              </div>
            </div>
          </Button>

          <Button
            variant="outline-secondary"
            size="lg"
            onClick={onCreateSeparate}
            className="text-start d-flex align-items-start gap-3 p-3"
          >
            <MaterialIcon name="schedule" size={24} className="mt-1" />
            <div>
              <div className="fw-bold mb-1">Create Separate Simulation</div>
              <div className="small text-muted">
                Create as an independent simulation that will start after the current queue
                completes
              </div>
            </div>
          </Button>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="link" onClick={onHide}>
          Cancel
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
