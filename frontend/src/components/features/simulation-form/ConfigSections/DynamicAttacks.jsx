import { useState } from 'react';
import { Form, Button, Card, Alert } from 'react-bootstrap';
import { SwitchField } from '../FormFields/SwitchField';
import { NumberField } from '../FormFields/NumberField';
import { SelectField } from '../FormFields/SelectField';

export function DynamicAttacks({ config, onChange }) {
  const [schedule, setSchedule] = useState(config.dynamic_attacks?.schedule || []);

  const handleEnabledChange = e => {
    const enabled = e.target.checked;
    onChange({
      target: {
        name: 'dynamic_attacks',
        value: { enabled, schedule: enabled ? schedule : [] },
      },
    });
  };

  const handleAddAttackPhase = () => {
    const newPhase = {
      start_round: 1,
      end_round: config.num_of_rounds || 10,
      selection_strategy: 'specific',
      client_ids: [0],
      attack_config: {
        type: 'label_flipping',
        params: { flip_fraction: 0.5, num_classes: 10 },
      },
    };
    const newSchedule = [...schedule, newPhase];
    setSchedule(newSchedule);
    onChange({
      target: {
        name: 'dynamic_attacks',
        value: { enabled: true, schedule: newSchedule },
      },
    });
  };

  const handleRemovePhase = index => {
    const newSchedule = schedule.filter((_, i) => i !== index);
    setSchedule(newSchedule);
    onChange({
      target: {
        name: 'dynamic_attacks',
        value: { enabled: config.dynamic_attacks?.enabled || false, schedule: newSchedule },
      },
    });
  };

  const handlePhaseChange = (index, field, value) => {
    const newSchedule = [...schedule];
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      newSchedule[index][parent][child] = value;
    } else {
      newSchedule[index][field] = value;
    }
    setSchedule(newSchedule);
    onChange({
      target: {
        name: 'dynamic_attacks',
        value: { enabled: config.dynamic_attacks?.enabled || false, schedule: newSchedule },
      },
    });
  };

  return (
    <>
      <SwitchField
        name="dynamic_attacks_enabled"
        label="Enable Dynamic Attacks"
        checked={config.dynamic_attacks?.enabled || false}
        onChange={handleEnabledChange}
        tooltip="Schedule attacks to occur during specific rounds (advanced feature)"
      />

      {config.dynamic_attacks?.enabled && (
        <>
          <Alert variant="info" className="small">
            Dynamic attacks allow you to schedule different attacks at different rounds. This is
            useful for testing defense adaptation over time.
          </Alert>

          {schedule.map((phase, index) => (
            <Card key={index} className="mb-3">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <span>Attack Phase {index + 1}</span>
                <Button variant="danger" size="sm" onClick={() => handleRemovePhase(index)}>
                  Remove
                </Button>
              </Card.Header>
              <Card.Body>
                <NumberField
                  name={`phase_${index}_start_round`}
                  label="Start Round"
                  value={phase.start_round}
                  onChange={e => handlePhaseChange(index, 'start_round', parseInt(e.target.value))}
                  min={1}
                  max={config.num_of_rounds}
                />

                <NumberField
                  name={`phase_${index}_end_round`}
                  label="End Round"
                  value={phase.end_round}
                  onChange={e => handlePhaseChange(index, 'end_round', parseInt(e.target.value))}
                  min={1}
                  max={config.num_of_rounds}
                />

                <SelectField
                  name={`phase_${index}_selection_strategy`}
                  label="Client Selection Strategy"
                  value={phase.selection_strategy}
                  onChange={e => handlePhaseChange(index, 'selection_strategy', e.target.value)}
                  options={['specific', 'random', 'all']}
                />

                {phase.selection_strategy === 'specific' && (
                  <Form.Group className="mb-3">
                    <Form.Label>Client IDs (comma-separated)</Form.Label>
                    <Form.Control
                      type="text"
                      value={phase.client_ids.join(', ')}
                      onChange={e =>
                        handlePhaseChange(
                          index,
                          'client_ids',
                          e.target.value.split(',').map(id => parseInt(id.trim()))
                        )
                      }
                      placeholder="e.g., 0, 1, 2"
                    />
                  </Form.Group>
                )}

                <SelectField
                  name={`phase_${index}_attack_type`}
                  label="Attack Type"
                  value={phase.attack_config.type}
                  onChange={e => handlePhaseChange(index, 'attack_config.type', e.target.value)}
                  options={['label_flipping', 'gaussian_noise']}
                />
              </Card.Body>
            </Card>
          ))}

          <Button variant="outline-primary" onClick={handleAddAttackPhase} className="w-100">
            + Add Attack Phase
          </Button>
        </>
      )}
    </>
  );
}
