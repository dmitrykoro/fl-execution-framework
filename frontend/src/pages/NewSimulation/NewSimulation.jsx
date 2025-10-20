import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Alert } from 'react-bootstrap';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { SimulationForm } from '@components/features/simulation-form/SimulationForm';
import { OutlineButton } from '@components/common/Button/OutlineButton';
import { ConfirmModal } from '@components/common/Modal/ConfirmModal';
import { QueueChoiceModal } from '@components/common/Modal/QueueChoiceModal';
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
  const [showQueueChoiceModal, setShowQueueChoiceModal] = useState(false);
  const [pendingConfig, setPendingConfig] = useState(null);
  const navigate = useNavigate();

  const validation = useConfigValidation(config);
  const { hasRunning, runningSimIds } = useRunningSimulation();

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

      if (sanitized.model_type === 'transformer') {
        sanitized.model_type = 'cnn';
      }
    }

    return sanitized;
  };

  const handleSubmit = async e => {
    e.preventDefault();

    if (hasRunning) {
      const sanitizedConfig = sanitizeConfig(config);
      setPendingConfig(sanitizedConfig);
      setShowQueueChoiceModal(true);
      return;
    }

    await submitSimulation(null);
  };

  const submitSimulation = async (addToQueue = null) => {
    setSubmitting(true);
    setError(null);
    setShowQueueChoiceModal(false);

    try {
      const configToSubmit = pendingConfig || sanitizeConfig(config);
      const response = await createSimulation(configToSubmit, addToQueue);
      const { simulation_id, queued } = response.data;
      localStorage.removeItem('simulation-draft');
      setPendingConfig(null);

      if (queued) {
        toast.success('Added to experiment queue!');
        navigate(`/queue/${simulation_id}`);
      } else {
        toast.success('Simulation created successfully!');
        navigate(`/simulations/${simulation_id}`);
      }
    } catch (err) {
      console.error('Failed to create simulation:', err);

      let errorMsg = 'An unexpected error occurred.';

      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;

        if (Array.isArray(detail)) {
          errorMsg = detail.map(e => e.msg || JSON.stringify(e)).join(', ');
        } else if (typeof detail === 'string') {
          errorMsg = detail;
        } else {
          errorMsg = detail.msg || JSON.stringify(detail);
        }
      }

      setError(errorMsg);
      toast.error(errorMsg, {
        duration: 5000,
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
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <i className="bi bi-exclamation-triangle me-2"></i>
              <strong>Simulation in progress</strong> - New simulations will queue automatically
            </div>
            <OutlineButton
              as={Link}
              to={`/queue/${runningSimIds[0]}`}
              className="btn-warning-action"
              size="sm"
            >
              View Queue Status
            </OutlineButton>
          </div>
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

      <QueueChoiceModal
        show={showQueueChoiceModal}
        onHide={() => {
          setShowQueueChoiceModal(false);
          setPendingConfig(null);
        }}
        onAddToQueue={() => submitSimulation(true)}
        onCreateSeparate={() => submitSimulation(false)}
      />
    </PageContainer>
  );
}
