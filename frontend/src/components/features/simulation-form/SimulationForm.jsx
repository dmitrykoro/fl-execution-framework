import { useState } from 'react';
import { Form, Button, Accordion, OverlayTrigger, Tooltip, Spinner } from 'react-bootstrap';
import { PresetSelector } from './PresetSelector';
import { CommonSettings } from './ConfigSections/CommonSettings';
import { AttackSettings } from './ConfigSections/AttackSettings';
import { DefenseSettings } from './ConfigSections/DefenseSettings';
import { TrainingSettings } from './ConfigSections/TrainingSettings';
import { FlowerSettings } from './ConfigSections/FlowerSettings';
import { TransformerSettings } from './ConfigSections/TransformerSettings';
import { LLMSettings } from './ConfigSections/LLMSettings';
import { OutputSettings } from './ConfigSections/OutputSettings';
import { DynamicAttacks } from './ConfigSections/DynamicAttacks';
import ValidationSummary from '@components/ValidationSummary';

export function SimulationForm({
  config,
  onConfigChange,
  selectedPreset,
  onPresetChange,
  onSubmit,
  isSubmitting,
  validation,
  error,
}) {
  const [activeSection, setActiveSection] = useState(['0']);

  // Helper function to get section preview text
  const getSectionPreview = section => {
    switch (section) {
      case 'common':
        return `${config.aggregation_strategy_keyword || 'fedavg'} • ${config.num_of_rounds || 0}r • ${config.num_of_clients || 0}c`;
      case 'attack':
        return `${config.num_of_malicious_clients || 0} malicious • ${config.attack_type || 'none'}`;
      case 'defense':
        return config.aggregation_strategy_keyword?.includes('pid')
          ? `Kp: ${config.Kp || 0} • Ki: ${config.Ki || 0} • Kd: ${config.Kd || 0}`
          : `Strategy: ${config.aggregation_strategy_keyword || 'fedavg'}`;
      case 'training':
        return `${config.num_of_client_epochs || 0} epochs • batch ${config.batch_size || 32}`;
      case 'flower':
        return `min_fit: ${config.min_fit_clients || 0} • min_eval: ${config.min_evaluate_clients || 0}`;
      case 'llm':
        return config.llm_enabled ? `Enabled • ${config.llm_provider || 'openai'}` : 'Disabled';
      case 'output':
        return `${config.preserve_dataset === 'true' ? 'Preserve dataset' : 'Clean after run'}`;
      case 'dynamic':
        return config.dynamic_attacks?.enabled
          ? `${config.dynamic_attacks.schedule?.length || 0} attack phases`
          : 'Disabled';
      default:
        return '';
    }
  };

  // Helper function to get tooltip explanations for section previews
  const getSectionPreviewTooltip = section => {
    switch (section) {
      case 'common':
        return (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>{config.aggregation_strategy_keyword || 'fedavg'}</strong> - Aggregation
              strategy
            </div>
            <div>
              <strong>{config.num_of_rounds || 0}r</strong> - Number of training rounds
            </div>
            <div>
              <strong>{config.num_of_clients || 0}c</strong> - Total number of clients
            </div>
          </div>
        );
      case 'attack':
        return (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>{config.num_of_malicious_clients || 0} malicious</strong> - Number of
              malicious clients
            </div>
            <div>
              <strong>{config.attack_type || 'none'}</strong> - Type of attack being simulated
            </div>
          </div>
        );
      case 'defense':
        return config.aggregation_strategy_keyword?.includes('pid') ? (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>PID Controller Gains:</strong>
            </div>
            <div>
              <strong>Kp: {config.Kp || 0}</strong> - Proportional gain (immediate response)
            </div>
            <div>
              <strong>Ki: {config.Ki || 0}</strong> - Integral gain (accumulated error)
            </div>
            <div>
              <strong>Kd: {config.Kd || 0}</strong> - Derivative gain (rate of change)
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>{config.aggregation_strategy_keyword || 'fedavg'}</strong> - Defense
              aggregation strategy
            </div>
          </div>
        );
      case 'training':
        return (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>{config.num_of_client_epochs || 0} epochs</strong> - Training epochs per
              client
            </div>
            <div>
              <strong>batch {config.batch_size || 32}</strong> - Training batch size
            </div>
          </div>
        );
      case 'flower':
        return (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>min_fit: {config.min_fit_clients || 0}</strong> - Minimum clients for training
            </div>
            <div>
              <strong>min_eval: {config.min_evaluate_clients || 0}</strong> - Minimum clients for
              evaluation
            </div>
          </div>
        );
      case 'llm':
        return config.llm_enabled ? (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>Enabled</strong> - LLM integration is active
            </div>
            <div>
              <strong>{config.llm_provider || 'openai'}</strong> - LLM provider for analysis
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>Disabled</strong> - LLM integration is turned off
            </div>
          </div>
        );
      case 'output':
        return (
          <div style={{ textAlign: 'left' }}>
            {config.preserve_dataset === 'true' ? (
              <div>
                <strong>Preserve dataset</strong> - Keep dataset files after simulation
              </div>
            ) : (
              <div>
                <strong>Clean after run</strong> - Remove dataset files after simulation
              </div>
            )}
          </div>
        );
      case 'dynamic':
        return config.dynamic_attacks?.enabled ? (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>{config.dynamic_attacks.schedule?.length || 0} attack phases</strong> - Number
              of scheduled attack phases
            </div>
            <div>Attacks that change during training rounds</div>
          </div>
        ) : (
          <div style={{ textAlign: 'left' }}>
            <div>
              <strong>Disabled</strong> - No dynamic attacks configured
            </div>
          </div>
        );
      default:
        return '';
    }
  };

  return (
    <Form onSubmit={onSubmit}>
      {/* Preset Selector */}
      <PresetSelector selectedPreset={selectedPreset} onPresetChange={onPresetChange} />

      {/* Display Name */}
      <Form.Group className="mb-4">
        <Form.Label>Simulation Name (Optional)</Form.Label>
        <Form.Control
          type="text"
          name="display_name"
          value={config.display_name || ''}
          onChange={onConfigChange}
          placeholder="Leave empty to use auto-generated ID"
        />
        <Form.Text className="text-muted">
          Give your simulation a memorable name, or use the auto-generated ID
        </Form.Text>
      </Form.Group>

      {/* Validation Summary */}
      {validation && <ValidationSummary validation={validation} />}

      {/* Error Alert */}
      {error && (
        <div className="alert alert-danger" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Quick Navigation Menu */}
      <div className="section-nav-menu mb-4">
        <div className="small text-muted mb-2">Jump to section:</div>
        <div className="d-flex flex-wrap gap-2">
          {[
            { key: '0', label: 'Common', section: 'common' },
            { key: '1', label: 'Attack', section: 'attack' },
            { key: '2', label: 'Defense', section: 'defense' },
            { key: '3', label: 'Training', section: 'training' },
            { key: '4', label: 'Flower', section: 'flower' },
            { key: '6', label: 'LLM', section: 'llm' },
            { key: '7', label: 'Output', section: 'output' },
            { key: '8', label: 'Dynamic', section: 'dynamic' },
          ].map(item => {
            const isActive = activeSection.includes(item.key);
            return (
              <button
                key={item.key}
                type="button"
                className={`btn btn-sm ${isActive ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => {
                  // Toggle the accordion: open if closed, close if open
                  if (activeSection.includes(item.key)) {
                    // Close the accordion
                    setActiveSection(activeSection.filter(key => key !== item.key));
                  } else {
                    // Open the accordion
                    setActiveSection([...activeSection, item.key]);
                  }
                  // Scroll to the section
                  const element = document.querySelector(`[data-section-key="${item.key}"]`);
                  element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Configuration Sections */}
      <Accordion
        defaultActiveKey={['0']}
        alwaysOpen
        className="mb-4"
        activeKey={activeSection}
        onSelect={setActiveSection}
      >
        <Accordion.Item eventKey="0" data-section-key="0">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Common Settings</span>
              {!activeSection.includes('0') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-common" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('common')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('common')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <CommonSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="1" data-section-key="1">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Attack Configuration</span>
              {!activeSection.includes('1') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-attack" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('attack')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('attack')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <AttackSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="2" data-section-key="2">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Defense Strategy Parameters</span>
              {!activeSection.includes('2') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-defense" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('defense')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('defense')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <DefenseSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="3" data-section-key="3">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Training Configuration</span>
              {!activeSection.includes('3') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-training" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('training')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('training')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <TrainingSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="4" data-section-key="4">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Flower Framework Settings</span>
              {!activeSection.includes('4') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-flower" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('flower')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('flower')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <FlowerSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        {config.model_type === 'transformer' && (
          <Accordion.Item eventKey="5" data-section-key="5">
            <Accordion.Header>Transformer Configuration</Accordion.Header>
            <Accordion.Body>
              <TransformerSettings config={config} onChange={onConfigChange} />
            </Accordion.Body>
          </Accordion.Item>
        )}

        <Accordion.Item eventKey="6" data-section-key="6">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>LLM Settings</span>
              {!activeSection.includes('6') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-llm" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('llm')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('llm')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <LLMSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="7" data-section-key="7">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Output Settings</span>
              {!activeSection.includes('7') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-output" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('output')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('output')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <OutputSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="8" data-section-key="8">
          <Accordion.Header>
            <div className="d-flex align-items-center justify-content-between w-100 me-3">
              <span>Dynamic Attacks (Advanced)</span>
              {!activeSection.includes('8') && (
                <OverlayTrigger
                  placement="left"
                  delay={{ show: 250, hide: 400 }}
                  overlay={
                    <Tooltip id="tooltip-dynamic" className="config-preview-tooltip">
                      {getSectionPreviewTooltip('dynamic')}
                    </Tooltip>
                  }
                >
                  <span className="text-muted small section-preview">
                    {getSectionPreview('dynamic')}
                  </span>
                </OverlayTrigger>
              )}
            </div>
          </Accordion.Header>
          <Accordion.Body>
            <DynamicAttacks config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>

      {/* Submit Button */}
      <div className="d-flex gap-2">
        <Button
          variant="primary"
          type="submit"
          disabled={isSubmitting || (validation && validation.errors.length > 0)}
          className="flex-grow-1"
        >
          {isSubmitting ? (
            <>
              <Spinner
                as="span"
                animation="border"
                size="sm"
                role="status"
                aria-hidden="true"
                className="me-2"
              />
              Creating Simulation...
            </>
          ) : (
            'Create Simulation'
          )}
        </Button>
      </div>
    </Form>
  );
}
