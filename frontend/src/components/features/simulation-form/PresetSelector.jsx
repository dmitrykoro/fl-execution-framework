import { Card, Row, Col } from 'react-bootstrap';
import { PRESETS } from '@constants/presets';

export function PresetSelector({ selectedPreset, onPresetChange }) {
  const handleKeyDown = (event, key) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onPresetChange(key);
    }
  };

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
              onKeyDown={e => handleKeyDown(e, key)}
              className={`preset-card ${selectedPreset === key ? 'border-primary' : ''}`}
              tabIndex={0}
              role="button"
              aria-pressed={selectedPreset === key}
              aria-label={`Select ${preset.name} preset`}
            >
              <Card.Body>
                <div className="preset-header">
                  <span className="preset-icon" aria-hidden="true">
                    {preset.icon}
                  </span>
                  <div className="preset-title-section">
                    <Card.Title className="h6">{preset.name}</Card.Title>
                    <Card.Subtitle className="text-muted small">{preset.subtitle}</Card.Subtitle>
                  </div>
                </div>
                <Card.Text className="small preset-description">{preset.description}</Card.Text>
                <div className="small text-muted preset-footer">
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
