import { Card, Row, Col } from 'react-bootstrap';
import { PRESETS } from '@constants/presets';

export function PresetSelector({ selectedPreset, onPresetChange }) {
  return (
    <div className="mb-4">
      <h5 className="mb-3">Quick Start Presets</h5>
      <p className="text-muted mb-3">
        Select a preset configuration to get started quickly, or scroll down to customize your own
        simulation.
      </p>
      <Row xs={1} md={2} lg={3} className="g-3">
        {Object.entries(PRESETS).map(([key, preset]) => (
          <Col key={key}>
            <Card
              onClick={() => onPresetChange(key)}
              className={`preset-card ${selectedPreset === key ? 'border-primary' : ''}`}
              style={{
                cursor: 'pointer',
                transition: 'all 0.2s',
                borderWidth: selectedPreset === key ? '2px' : '1px',
              }}
            >
              <Card.Body>
                <div className="d-flex align-items-start gap-2 mb-2">
                  <span style={{ fontSize: '2rem' }}>{preset.icon}</span>
                  <div>
                    <Card.Title className="mb-0 h6">{preset.name}</Card.Title>
                    <Card.Subtitle className="text-muted small">{preset.subtitle}</Card.Subtitle>
                  </div>
                </div>
                <Card.Text className="small mb-2">{preset.description}</Card.Text>
                <div className="small text-muted">
                  <strong>Est. time:</strong> {preset.estimatedTime}
                </div>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}
