import { Form, Button, Accordion } from 'react-bootstrap';
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
          Give your simulation a memorable name, or leave blank to use the auto-generated ID
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

      {/* Configuration Sections */}
      <Accordion defaultActiveKey={['0']} alwaysOpen className="mb-4">
        <Accordion.Item eventKey="0">
          <Accordion.Header>Common Settings</Accordion.Header>
          <Accordion.Body>
            <CommonSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="1">
          <Accordion.Header>Attack Configuration</Accordion.Header>
          <Accordion.Body>
            <AttackSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="2">
          <Accordion.Header>Defense Strategy Parameters</Accordion.Header>
          <Accordion.Body>
            <DefenseSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="3">
          <Accordion.Header>Training Configuration</Accordion.Header>
          <Accordion.Body>
            <TrainingSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="4">
          <Accordion.Header>Flower Framework Settings</Accordion.Header>
          <Accordion.Body>
            <FlowerSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        {config.model_type === 'transformer' && (
          <Accordion.Item eventKey="5">
            <Accordion.Header>Transformer Configuration</Accordion.Header>
            <Accordion.Body>
              <TransformerSettings config={config} onChange={onConfigChange} />
            </Accordion.Body>
          </Accordion.Item>
        )}

        <Accordion.Item eventKey="6">
          <Accordion.Header>LLM Settings</Accordion.Header>
          <Accordion.Body>
            <LLMSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="7">
          <Accordion.Header>Output Settings</Accordion.Header>
          <Accordion.Body>
            <OutputSettings config={config} onChange={onConfigChange} />
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="8">
          <Accordion.Header>Dynamic Attacks (Advanced)</Accordion.Header>
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
          {isSubmitting ? 'Creating Simulation...' : 'Create Simulation'}
        </Button>
      </div>
    </Form>
  );
}
