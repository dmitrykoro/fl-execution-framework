import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Tabs, Tab, Alert, Button, Spinner } from 'react-bootstrap';
import { SharedConfigEditor } from './SharedConfigEditor';
import { StrategyVariationList } from './StrategyVariationList';
import { initialConfig } from '@constants/initialConfig';
import { useConfigValidation } from '@hooks/useConfigValidation';
import { useMultiSimulation } from '@hooks/useMultiSimulation';
import { toast } from 'sonner';

export function QueueBuilder() {
  const [activeTab, setActiveTab] = useState('shared-config');
  const [sharedConfig, setSharedConfig] = useState(initialConfig);
  const [strategyVariations, setStrategyVariations] = useState([]);

  const navigate = useNavigate();
  const validation = useConfigValidation(sharedConfig);
  const { createMultiSimulation, isSubmitting } = useMultiSimulation();

  const handleConfigChange = e => {
    const { name, value, type } = e.target;
    let finalValue = value;

    if (type === 'number') {
      finalValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    }

    setSharedConfig(prev => ({ ...prev, [name]: finalValue }));
  };

  const buildMultiSimConfig = () => {
    const {
      aggregation_strategy_keyword: _aggregation_strategy_keyword,
      num_of_malicious_clients: _num_of_malicious_clients,
      num_krum_selections: _num_krum_selections,
      remove_clients: _remove_clients,
      ...shared_settings
    } = sharedConfig;

    const simulation_strategies = strategyVariations.map(variation => {
      const { id: _id, name: _name, ...strategyParams } = variation;
      return strategyParams;
    });

    return {
      shared_settings,
      simulation_strategies,
    };
  };

  const handleSubmit = async () => {
    if (strategyVariations.length < 2) {
      toast.error('At least 2 strategies are required');
      return;
    }

    if (validation && validation.errors.length > 0) {
      toast.error('Please fix validation errors in shared config');
      setActiveTab('shared-config');
      return;
    }

    try {
      const multiSimConfig = buildMultiSimConfig();
      const response = await createMultiSimulation(multiSimConfig);
      const { simulation_id } = response.data;

      toast.success(`Experiment queue created! (${strategyVariations.length} strategies)`);
      navigate(`/queue/${simulation_id}`);
    } catch (err) {
      console.error('Failed to create experiment queue:', err);
      const errorMsg = err.response?.data?.detail || 'An unexpected error occurred.';
      toast.error(errorMsg, { duration: 5000 });
    }
  };

  const handleDownloadPreview = () => {
    const config = buildMultiSimConfig();
    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'experiment-queue-config.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success('Configuration downloaded');
  };

  const previewConfig = buildMultiSimConfig();
  const canSubmit =
    strategyVariations.length >= 2 && (!validation || validation.errors.length === 0);

  return (
    <div className="queue-builder">
      <Alert variant="info" className="mb-4">
        <i className="bi bi-info-circle me-2"></i>
        <strong>Experiment Queue</strong> allows you to define a shared baseline configuration and
        create multiple strategy variations that will execute sequentially.
      </Alert>

      <Tabs
        activeKey={activeTab}
        onSelect={k => setActiveTab(k)}
        className="mb-4"
        variant="underline"
      >
        <Tab eventKey="shared-config" title="1. Shared Config">
          <div className="py-3">
            <SharedConfigEditor
              config={sharedConfig}
              onConfigChange={handleConfigChange}
              validation={validation}
            />
          </div>
        </Tab>

        <Tab eventKey="variations" title="2. Strategy Variations">
          <div className="py-3">
            <StrategyVariationList
              variations={strategyVariations}
              onChange={setStrategyVariations}
              numOfClients={sharedConfig.num_of_clients}
            />
          </div>
        </Tab>

        <Tab eventKey="preview" title="3. Preview & Submit">
          <div className="py-3">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h5 className="mb-1">Configuration Preview</h5>
                <p className="text-muted small mb-0">
                  Review your experiment queue configuration before submission
                </p>
              </div>
              <Button variant="outline-secondary" size="sm" onClick={handleDownloadPreview}>
                Download JSON
              </Button>
            </div>

            {strategyVariations.length === 0 ? (
              <Alert variant="warning">
                <i className="bi bi-exclamation-triangle me-2"></i>
                No strategies defined. Please add at least 2 strategies in the previous tab.
              </Alert>
            ) : (
              <>
                <div className="config-preview-box mb-4">
                  <pre className="bg-dark text-light p-3 rounded">
                    <code>{JSON.stringify(previewConfig, null, 2)}</code>
                  </pre>
                </div>

                <div className="d-flex justify-content-between align-items-center">
                  <div className="text-muted small">
                    <i className="bi bi-info-circle me-2"></i>
                    {strategyVariations.length} strategies will execute sequentially
                  </div>
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={handleSubmit}
                    disabled={!canSubmit || isSubmitting}
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
                        Creating Queue...
                      </>
                    ) : (
                      'Submit Experiment Queue'
                    )}
                  </Button>
                </div>
              </>
            )}
          </div>
        </Tab>
      </Tabs>
    </div>
  );
}
