import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { SimulationForm } from '@components/features/simulation-form/SimulationForm';
import { OutlineButton } from '@components/common/Button/OutlineButton';
import { ConfirmModal } from '@components/common/Modal/ConfirmModal';
import { createSimulation } from '@api';
import { useConfigValidation } from '@hooks/useConfigValidation';
import { initialConfig } from '@constants/initialConfig';
import { PRESETS } from '@constants/presets';
import { useToast } from '@contexts/ToastContext';

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
  const [draftSaved, setDraftSaved] = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);
  const navigate = useNavigate();
  const { showSuccess } = useToast();

  const validation = useConfigValidation(config);

  // Auto-save to localStorage when config changes
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      localStorage.setItem('simulation-draft', JSON.stringify(config));
      setDraftSaved(true);
      setTimeout(() => setDraftSaved(false), 2000);
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

  const handleSubmit = async e => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const response = await createSimulation(config);
      const { simulation_id } = response.data;
      localStorage.removeItem('simulation-draft');
      navigate(`/simulations/${simulation_id}`);
    } catch (err) {
      console.error('Failed to create simulation:', err);
      setError(err.response?.data?.detail || 'An unexpected error occurred.');
      setSubmitting(false);
    }
  };

  const handleClearDraft = () => {
    setShowClearModal(true);
  };

  const confirmClearDraft = () => {
    localStorage.removeItem('simulation-draft');
    setConfig(initialConfig);
    setSelectedPreset(null);
    setShowClearModal(false);
    showSuccess('Draft cleared and reset to defaults');
  };

  return (
    <PageContainer>
      <PageHeader title="New Simulation">
        {draftSaved && <span className="text-muted small">Draft saved...</span>}
        <OutlineButton onClick={handleClearDraft} variant="outline-secondary">
          Clear Draft
        </OutlineButton>
      </PageHeader>

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
        title="Clear Draft"
        message="This will reset all fields to default values. Continue?"
        variant="warning"
        onConfirm={confirmClearDraft}
        onCancel={() => setShowClearModal(false)}
      />
    </PageContainer>
  );
}
