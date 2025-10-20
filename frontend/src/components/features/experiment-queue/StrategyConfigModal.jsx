import { useState } from 'react';
import { Modal, Button, Form, Row, Col } from 'react-bootstrap';

export function StrategyConfigModal({ show, onHide, strategy, onSave }) {
  const [editedConfig, setEditedConfig] = useState(strategy?.config || {});

  const handleChange = e => {
    const { name, value, type } = e.target;
    let finalValue = value;

    if (type === 'number') {
      finalValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    }

    setEditedConfig(prev => ({ ...prev, [name]: finalValue }));
  };

  const handleSave = () => {
    onSave(strategy.index, editedConfig);
    onHide();
  };

  if (!strategy) return null;

  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>
          <i className="bi bi-gear me-2"></i>
          Strategy {strategy.index + 1} Configuration
        </Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Form>
          <Row className="mb-3">
            <Col md={6}>
              <Form.Group>
                <Form.Label>Aggregation Strategy</Form.Label>
                <Form.Select
                  name="aggregation_strategy_keyword"
                  value={editedConfig.aggregation_strategy_keyword || 'fedavg'}
                  onChange={handleChange}
                >
                  <option value="fedavg">FedAvg</option>
                  <option value="krum">Krum</option>
                  <option value="trimmed_mean">Trimmed Mean</option>
                  <option value="median">Median</option>
                  <option value="rfa">RFA</option>
                  <option value="trust">TRUST</option>
                </Form.Select>
              </Form.Group>
            </Col>

            <Col md={6}>
              <Form.Group>
                <Form.Label>Malicious Clients</Form.Label>
                <Form.Control
                  type="number"
                  name="num_of_malicious_clients"
                  value={editedConfig.num_of_malicious_clients || 0}
                  onChange={handleChange}
                  min="0"
                />
              </Form.Group>
            </Col>
          </Row>

          {(editedConfig.aggregation_strategy_keyword === 'krum' ||
            editedConfig.aggregation_strategy_keyword === 'trimmed_mean') && (
            <Row className="mb-3">
              <Col md={6}>
                <Form.Group>
                  <Form.Label>
                    {editedConfig.aggregation_strategy_keyword === 'krum'
                      ? 'Krum Selections (k)'
                      : 'Clients to Remove'}
                  </Form.Label>
                  <Form.Control
                    type="number"
                    name={
                      editedConfig.aggregation_strategy_keyword === 'krum'
                        ? 'num_krum_selections'
                        : 'remove_clients'
                    }
                    value={
                      editedConfig.aggregation_strategy_keyword === 'krum'
                        ? editedConfig.num_krum_selections || 1
                        : editedConfig.remove_clients || 0
                    }
                    onChange={handleChange}
                    min="0"
                  />
                </Form.Group>
              </Col>
            </Row>
          )}

          <div className="bg-light p-3 rounded">
            <h6 className="mb-2">Current Configuration</h6>
            <pre className="mb-0 small">
              <code>{JSON.stringify(editedConfig, null, 2)}</code>
            </pre>
          </div>
        </Form>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Cancel
        </Button>
        <Button variant="primary" onClick={handleSave}>
          <i className="bi bi-check-lg me-2"></i>
          Save Changes
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
