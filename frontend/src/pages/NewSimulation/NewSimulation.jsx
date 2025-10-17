import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Alert } from 'react-bootstrap';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { SimulationForm } from '@components/features/simulation-form/SimulationForm';
import { OutlineButton } from '@components/common/Button/OutlineButton';
import { ConfirmModal } from '@components/common/Modal/ConfirmModal';
import { createSimulation } from '@api';
import { useConfigValidation } from '@hooks/useConfigValidation';
import { useRunningSimulation } from '@hooks/useRunningSimulation';
import { initialConfig } from '@constants/initialConfig';
import { PRESETS } from '@constants/presets';
import { toast } from 'sonner';

export function NewSimulation() {
  const [config, setConfig] = useState(() => {
    const savedDraft = localStorage.getItem('simulation-draft');
    if (savedDraft) {
      try {
        return JSON.parse(savedDraft);
      } catch (e) {
        console.error('Failed to parse saved draft:', e);
        return initialConfig;
      }
    }
    return initialConfig;
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [showClearModal, setShowClearModal] = useState(false);
  const navigate = useNavigate();

  const validation = useConfigValidation(config);
  const { hasRunning } = useRunningSimulation();

  // Auto-save to localStorage when config changes
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      localStorage.setItem('simulation-draft', JSON.stringify(config));
    }, 1000);

    return () => clearTimeout(timeoutId);
  }, [config]);

  const handleConfigChange = e => {
    const { name, value, type } = e.target;
    let finalValue = value;

    if (type === 'number') {
      finalValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    }

    setConfig(prev => ({ ...prev, [name]: finalValue }));
  };

  const handlePresetChange = presetKey => {
    setSelectedPreset(presetKey);
    if (presetKey && PRESETS[presetKey]) {
      const preset = PRESETS[presetKey];
      setConfig(prev => ({
        ...prev,
        ...preset.config,
        display_name: preset.name,
      }));
    }
  };

  /**
   * Sanitize config by removing incompatible fields based on dataset_source.
   * Prevents HuggingFace configs from including local dataset fields and vice versa.
   */
  const sanitizeConfig = config => {
    const sanitized = { ...config };

    if (config.dataset_source === 'huggingface') {
      delete sanitized.dataset_keyword;
    } else {
      delete sanitized.hf_dataset_name;
      delete sanitized.transformer_model;
      delete sanitized.max_seq_length;
      delete sanitized.text_column;
      delete sanitized.text2_column;
      delete sanitized.label_column;
      delete sanitized.use_lora;
      delete sanitized.lora_rank;
      delete sanitized.partitioning_strategy;
      delete sanitized.partitioning_params;

      // Transformers only supported with HuggingFace datasets
      if (sanitized.model_type === 'transformer') {
        sanitized.model_type = 'cnn';
      }
    }

    return sanitized;
  };

  const handleSubmit = async e => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const sanitizedConfig = sanitizeConfig(config);
      const response = await createSimulation(sanitizedConfig);
      const { simulation_id } = response.data;
      localStorage.removeItem('simulation-draft');
      toast.success('Simulation created successfully!');
      navigate(`/simulations/${simulation_id}`);
    } catch (err) {
      console.error('Failed to create simulation:', err);
      const errorMsg = err.response?.data?.detail || 'An unexpected error occurred.';
      setError(errorMsg);
      toast.error(errorMsg, {
        duration: 5000,
        action: {
          label: 'Retry',
          onClick: () => handleSubmit(e),
        },
      });
      setSubmitting(false);
    }
  };

  const handleResetConfig = () => {
    setShowClearModal(true);
  };

  const confirmResetConfig = () => {
    localStorage.removeItem('simulation-draft');
    setConfig(initialConfig);
    setSelectedPreset(null);
    setShowClearModal(false);
    toast.success('Configuration reset to defaults');
  };

  const handleDownloadJSON = () => {
    const sanitizedConfig = sanitizeConfig(config);
    const dataStr = JSON.stringify(sanitizedConfig, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${config.display_name || 'simulation'}-config.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Configuration downloaded');
  };

  const handleUploadJSON = e => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      try {
        const uploadedConfig = JSON.parse(event.target.result);
        setConfig(prev => ({ ...prev, ...uploadedConfig }));
        setSelectedPreset(null);
        toast.success('Configuration loaded from file');
      } catch (err) {
        console.error('Failed to parse JSON:', err);
        toast.error('Invalid JSON file');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <PageContainer>
      <PageHeader title="New Simulation">
        <div className="d-flex gap-2">
          <input
            type="file"
            accept=".json"
            onChange={handleUploadJSON}
            style={{ display: 'none' }}
            id="upload-json-input"
          />
          <OutlineButton
            onClick={() => document.getElementById('upload-json-input').click()}
            variant="outline-secondary"
          >
            Upload JSON
          </OutlineButton>
          <OutlineButton onClick={handleDownloadJSON} variant="outline-secondary">
            Download JSON
          </OutlineButton>
          <OutlineButton onClick={handleResetConfig} variant="outline-secondary">
            Reset Configuration
          </OutlineButton>
        </div>
      </PageHeader>

      {hasRunning && (
        <Alert variant="warning" className="mb-4">
          <i className="bi bi-exclamation-triangle me-2"></i>
          <strong>Simulation currently running</strong> - You can still create a simulation, but it
          will queue and start automatically after the current one completes. For better control
          over multiple experiments, use the Experiment Queue feature.
        </Alert>
      )}

      <SimulationForm
        config={config}
        onConfigChange={handleConfigChange}
        selectedPreset={selectedPreset}
        onPresetChange={handlePresetChange}
        onSubmit={handleSubmit}
        isSubmitting={submitting}
        validation={validation}
        error={error}
      />

      <ConfirmModal
        show={showClearModal}
        title="Reset Configuration"
        message="This will reset all fields to default values. Continue?"
        variant="warning"
        onConfirm={confirmResetConfig}
        onCancel={() => setShowClearModal(false)}
      />
    </PageContainer>
  );
}
