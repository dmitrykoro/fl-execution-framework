import { Form } from 'react-bootstrap';
import { NumberField } from '../FormFields/NumberField';
import { SelectField } from '../FormFields/SelectField';
import { SwitchField } from '../FormFields/SwitchField';

export function LLMSettings({ config, onChange }) {
  const needsLlmParams = config.use_llm === 'true';
  const needsMLMParams = needsLlmParams && config.llm_task === 'mlm';
  const needsLoRAParams = needsLlmParams && config.llm_finetuning === 'lora';

  return (
    <>
      <SwitchField
        name="use_llm"
        label="Use LLM"
        checked={config.use_llm === 'true'}
        onChange={e =>
          onChange({ target: { name: 'use_llm', value: e.target.checked ? 'true' : 'false' } })
        }
        tooltip="Enable Large Language Model fine-tuning instead of traditional CNN/MLP models"
      />

      {needsLlmParams && (
        <>
          <Form.Group className="mb-3">
            <Form.Label>LLM Model</Form.Label>
            <Form.Control
              type="text"
              name="llm_model"
              value={config.llm_model}
              onChange={onChange}
              placeholder="e.g., microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
            />
          </Form.Group>

          <SelectField
            name="llm_finetuning"
            label="Fine-tuning Method"
            value={config.llm_finetuning}
            onChange={onChange}
            options={['lora', 'full']}
            tooltip="LoRA is parameter-efficient, full fine-tuning updates all parameters"
          />

          <SelectField
            name="llm_task"
            label="Task"
            value={config.llm_task}
            onChange={onChange}
            options={['mlm', 'classification']}
            tooltip="Masked Language Modeling (MLM) or text classification"
          />

          <NumberField
            name="llm_chunk_size"
            label="Chunk Size"
            value={config.llm_chunk_size}
            onChange={onChange}
            min={64}
            tooltip="Maximum sequence length for tokenization"
          />

          {needsMLMParams && (
            <NumberField
              name="mlm_probability"
              label="MLM Probability"
              value={config.mlm_probability}
              onChange={onChange}
              step={0.01}
              min={0}
              max={1}
              tooltip="Probability of masking each token for MLM task (typically 0.15)"
            />
          )}

          {needsLoRAParams && (
            <>
              <NumberField
                name="lora_rank"
                label="LoRA Rank"
                value={config.lora_rank}
                onChange={onChange}
                min={1}
                tooltip="Rank of LoRA matrices. Lower = fewer parameters (8-64 typical)"
              />

              <NumberField
                name="lora_alpha"
                label="LoRA Alpha"
                value={config.lora_alpha}
                onChange={onChange}
                min={1}
                tooltip="LoRA scaling parameter (typically 2x rank)"
              />

              <NumberField
                name="lora_dropout"
                label="LoRA Dropout"
                value={config.lora_dropout}
                onChange={onChange}
                step={0.01}
                min={0}
                max={1}
                tooltip="Dropout probability for LoRA layers (0.1 typical)"
              />

              <Form.Group className="mb-3">
                <Form.Label>LoRA Target Modules</Form.Label>
                <Form.Control
                  type="text"
                  name="lora_target_modules"
                  value={
                    Array.isArray(config.lora_target_modules)
                      ? config.lora_target_modules.join(', ')
                      : config.lora_target_modules
                  }
                  onChange={e =>
                    onChange({
                      target: {
                        name: 'lora_target_modules',
                        value: e.target.value.split(',').map(s => s.trim()),
                      },
                    })
                  }
                  placeholder="e.g., query, value"
                />
                <Form.Text className="text-muted">
                  Comma-separated list of modules to apply LoRA (e.g., query, value, key)
                </Form.Text>
              </Form.Group>
            </>
          )}
        </>
      )}
    </>
  );
}
