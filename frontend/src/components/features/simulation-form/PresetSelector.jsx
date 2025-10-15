import { Card, Row, Col, Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { motion } from 'framer-motion';
import { PRESETS } from '@constants/presets';

export function PresetSelector({ selectedPreset, onPresetChange }) {
  const handleKeyDown = (event, key) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onPresetChange(key);
    }
  };

  const requiresModelDownload = preset => {
    return preset.config.model_type === 'transformer';
  };

  const getDatasetInfo = preset => {
    const { config } = preset;
    const isLocal = config.dataset_source === 'local';
    const sourceEmoji = isLocal ? '📁' : '☁️';

    // Determine dataset name
    let datasetName;
    if (config.hf_dataset_name) {
      datasetName = config.hf_dataset_name.split('/').pop().toUpperCase();
    } else if (config.dataset_keyword) {
      const parts = config.dataset_keyword.split('_');
      datasetName = parts[0].toUpperCase();
      if (parts.length > 1) {
        datasetName += ` (${parts.slice(1).join('-').toUpperCase()})`;
      }
    } else {
      datasetName = 'Custom';
    }

    // Determine data type
    const isText = config.model_type === 'transformer' || config.text_column;
    const typeEmoji = isText ? '📝' : '🖼️';
    const dataType = isText ? 'Text' : 'Image';

    return { sourceEmoji, datasetName, typeEmoji, dataType };
  };

  const getBadgeClassName = tag => {
    const tagLower = tag.toLowerCase();
    if (tagLower === 'beginner') return 'bg-secondary difficulty-beginner';
    if (tagLower === 'intermediate') return 'bg-secondary difficulty-intermediate';
    if (tagLower === 'advanced') return 'bg-secondary difficulty-advanced';
    return 'bg-secondary';
  };

  // Animation variants for staggered list
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.1,
      },
    },
  };

  const cardVariants = {
    hidden: {
      opacity: 0,
      y: 20,
      scale: 0.95,
    },
    show: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 24,
      },
    },
  };

  return (
    <div className="mb-4">
      <h3 className="mb-3 preset-selector-heading">Quick Start Presets</h3>
      <p className="text-muted mb-3">
        Select a preset configuration to get started quickly, or scroll down to customize your own
        simulation.
      </p>
      <Row
        as={motion.div}
        xs={1}
        md={2}
        lg={3}
        className="g-3 align-items-stretch"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        {Object.entries(PRESETS).map(([key, preset]) => (
          <Col key={key} className="d-flex" as={motion.div} variants={cardVariants}>
            <motion.div
              className="w-100"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 400, damping: 17 }}
            >
              <Card
                onClick={() => onPresetChange(key)}
                onKeyDown={e => handleKeyDown(e, key)}
                className={`preset-card h-100 ${selectedPreset === key ? 'border-primary' : ''}`}
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
                  {preset.tags && (
                    <div className="mb-2">
                      {preset.tags.map(tag => (
                        <Badge key={tag} className={`me-1 ${getBadgeClassName(tag)}`}>
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div className="small text-muted preset-footer">
                    <div>
                      <strong>Est. time:</strong> {preset.estimatedTime}
                      {requiresModelDownload(preset) && (
                        <OverlayTrigger
                          placement="top"
                          overlay={
                            <Tooltip>
                              First-time run includes model download overhead. Subsequent runs are
                              faster.
                            </Tooltip>
                          }
                        >
                          <span
                            className="ms-2"
                            style={{ cursor: 'help', color: '#ffc107' }}
                            aria-label="Model download warning"
                          >
                            ⚠️
                          </span>
                        </OverlayTrigger>
                      )}
                    </div>
                    <div>
                      <strong>Dataset:</strong> {getDatasetInfo(preset).sourceEmoji}{' '}
                      {getDatasetInfo(preset).datasetName} {getDatasetInfo(preset).typeEmoji}{' '}
                      {getDatasetInfo(preset).dataType}
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </motion.div>
          </Col>
        ))}
      </Row>
    </div>
  );
}
