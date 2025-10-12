import { Form } from 'react-bootstrap';
import { NumberField } from '../FormFields/NumberField';
import { SwitchField } from '../FormFields/SwitchField';

export function TransformerSettings({ config, onChange }) {
  const useLoRA = config.use_lora !== false;

  return (
    <>
      <Form.Group className="mb-3">
        <Form.Label>Transformer Model</Form.Label>
        <Form.Control
          type="text"
          name="transformer_model"
          value={config.transformer_model || ''}
          onChange={onChange}
          placeholder="e.g., distilbert-base-uncased"
        />
        <Form.Text className="text-muted">
          HuggingFace model identifier (e.g., distilbert-base-uncased, roberta-base,
          bert-base-uncased)
        </Form.Text>
      </Form.Group>

      <NumberField
        name="max_seq_length"
        label="Max Sequence Length"
        value={config.max_seq_length || 128}
        onChange={onChange}
        min={32}
        max={512}
        tooltip="Maximum sequence length for tokenization. Longer = more context but slower training. 128-256 typical."
      />

      <Form.Group className="mb-3">
        <Form.Label>Text Column</Form.Label>
        <Form.Control
          type="text"
          name="text_column"
          value={config.text_column || ''}
          onChange={onChange}
          placeholder="e.g., text, sentence"
        />
        <Form.Text className="text-muted">
          Name of the text column in the dataset (e.g., 'text', 'sentence', 'premise')
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Second Text Column (Optional)</Form.Label>
        <Form.Control
          type="text"
          name="text2_column"
          value={config.text2_column || ''}
          onChange={onChange}
          placeholder="e.g., hypothesis (for NLI tasks)"
        />
        <Form.Text className="text-muted">
          For sentence pair tasks like NLI (e.g., 'hypothesis', 'question'). Leave empty for
          single-text classification.
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>Label Column</Form.Label>
        <Form.Control
          type="text"
          name="label_column"
          value={config.label_column || ''}
          onChange={onChange}
          placeholder="e.g., label"
        />
        <Form.Text className="text-muted">
          Name of the label column in the dataset (typically 'label')
        </Form.Text>
      </Form.Group>

      <SwitchField
        name="use_lora"
        label="Use LoRA"
        checked={useLoRA}
        onChange={e => onChange({ target: { name: 'use_lora', value: e.target.checked } })}
        tooltip="LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning, reducing trainable parameters by 95-99% while maintaining accuracy"
      />

      {useLoRA && (
        <NumberField
          name="lora_rank"
          label="LoRA Rank"
          value={config.lora_rank || 8}
          onChange={onChange}
          min={1}
          max={64}
          tooltip="Rank of LoRA adaptation matrices. Lower = fewer parameters but less flexibility. 4-16 typical, 8 is a good default."
        />
      )}
    </>
  );
}
