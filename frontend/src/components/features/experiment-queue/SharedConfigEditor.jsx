import { Alert, Form, Row, Col } from 'react-bootstrap';

export function SharedConfigEditor({ config, onConfigChange }) {
  return (
    <div className="shared-config-editor">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h5 className="mb-1">Shared Settings</h5>
          <p className="text-muted small mb-0">
            Configuration values that apply to all strategies in this experiment queue
          </p>
        </div>
      </div>

      <Form>
        <Row>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-arrow-repeat me-2"></i>
                Number of Rounds
              </Form.Label>
              <Form.Control
                type="number"
                name="num_of_rounds"
                value={config.num_of_rounds || 10}
                onChange={onConfigChange}
                min={1}
              />
              <Form.Text className="text-muted">How many training rounds to execute</Form.Text>
            </Form.Group>
          </Col>

          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-people me-2"></i>
                Number of Clients
              </Form.Label>
              <Form.Control
                type="number"
                name="num_of_clients"
                value={config.num_of_clients || 5}
                onChange={onConfigChange}
                min={1}
              />
              <Form.Text className="text-muted">Total clients in the federation</Form.Text>
            </Form.Group>
          </Col>
        </Row>

        <Row>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-database me-2"></i>
                Dataset
              </Form.Label>
              <Form.Select
                name="dataset_keyword"
                value={config.dataset_keyword || 'femnist_iid'}
                onChange={onConfigChange}
              >
                <option value="femnist_iid">FEMNIST (IID)</option>
                <option value="femnist_niid">FEMNIST (Non-IID)</option>
                <option value="its">ITS</option>
                <option value="pneumoniamnist">PneumoniaMNIST</option>
                <option value="bloodmnist">BloodMNIST</option>
                <option value="flair">FLAIR</option>
                <option value="medquad">MedQuAD</option>
                <option value="lung_photos">Lung Photos</option>
              </Form.Select>
              <Form.Text className="text-muted">Dataset for training</Form.Text>
            </Form.Group>
          </Col>

          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-layers me-2"></i>
                Model Type
              </Form.Label>
              <Form.Select
                name="model_type"
                value={config.model_type || 'cnn'}
                onChange={onConfigChange}
              >
                <option value="cnn">CNN</option>
                <option value="transformer">Transformer</option>
              </Form.Select>
              <Form.Text className="text-muted">Neural network architecture</Form.Text>
            </Form.Group>
          </Col>
        </Row>

        <Row>
          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-calendar-week me-2"></i>
                Client Epochs
              </Form.Label>
              <Form.Control
                type="number"
                name="num_of_client_epochs"
                value={config.num_of_client_epochs || 1}
                onChange={onConfigChange}
                min={1}
              />
              <Form.Text className="text-muted">Training epochs per client per round</Form.Text>
            </Form.Group>
          </Col>

          <Col md={6}>
            <Form.Group className="mb-3">
              <Form.Label>
                <i className="bi bi-stack me-2"></i>
                Batch Size
              </Form.Label>
              <Form.Control
                type="number"
                name="batch_size"
                value={config.batch_size || 20}
                onChange={onConfigChange}
                min={1}
              />
              <Form.Text className="text-muted">Training batch size</Form.Text>
            </Form.Group>
          </Col>
        </Row>

        <Alert variant="success" className="mt-3">
          <i className="bi bi-info-circle me-2"></i>
          <strong>Note:</strong> Strategy-specific settings (aggregation method, attack
          configuration, defense parameters) will be defined per-strategy in the next tab.
        </Alert>
      </Form>
    </div>
  );
}
