import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Alert, Tabs, Tab, Card, ProgressBar, Spinner } from 'react-bootstrap';
import { PageContainer } from '@components/layout/PageContainer';
import { SimulationHeader } from '@components/features/simulation-details/SimulationHeader';
import { InsightsTab } from '@components/features/simulation-details/InsightsTab';
import { MetricsTab } from '@components/features/simulation-details/MetricsTab/MetricsTab';
import { ConfigTab } from '@components/features/simulation-details/ConfigTab/ConfigTab';
import { PlotsTab } from '@components/features/simulation-details/PlotsTab/PlotsTab';
import { useSimulationDetails } from '@hooks/useSimulationDetails';
import { useCSVData } from '@hooks/useCSVData';
import { createSimulation, stopSimulation } from '@api/endpoints/simulations';
import { useToast } from '@contexts/ToastContext';

export function SimulationDetails() {
  const { simulationId } = useParams();
  const navigate = useNavigate();
  const { details, status, loading, error } = useSimulationDetails(simulationId);
  const { csvData } = useCSVData(simulationId, details?.result_files);
  const [isCloning, setIsCloning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const { showSuccess, showError } = useToast();

  const handleRunAgain = async () => {
    setIsCloning(true);
    try {
      const configToSend = details.config.shared_settings || details.config;
      const response = await createSimulation(configToSend);
      navigate(`/simulations/${response.data.simulation_id}`);
    } catch (err) {
      console.error('Failed to clone simulation:', err);
      alert('Failed to start new simulation');
      setIsCloning(false);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);
    try {
      await stopSimulation(simulationId);
      showSuccess('Simulation stopped successfully');
    } catch (err) {
      console.error('Failed to stop simulation:', err);
      showError(`Failed to stop: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsStopping(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <h1>Simulation Details: {simulationId}</h1>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </PageContainer>
    );
  }

  if (error) {
    return (
      <PageContainer>
        <h1>Simulation Details: {simulationId}</h1>
        <Alert variant="danger">{error}</Alert>
      </PageContainer>
    );
  }

  if (!details || !details.config) {
    return null;
  }

  const displayStatus = status || details.status;
  const csvFiles = details.result_files?.filter(file => file.endsWith('.csv')) || [];
  const cfg = details.config?.shared_settings || details.config;

  return (
    <PageContainer>
      <SimulationHeader
        simulation={{ ...details, id: simulationId, status: displayStatus }}
        onRunAgain={handleRunAgain}
        isCloning={isCloning}
        onStop={handleStop}
        isStopping={isStopping}
      />

      {displayStatus === 'running' && (
        <Card className="mb-3">
          <Card.Body>
            <div className="d-flex align-items-center gap-3 mb-2">
              <h6 className="mb-0 flex-grow-1">Simulation in Progress</h6>
              <Spinner animation="border" size="sm" />
            </div>
            <ProgressBar animated now={100} variant="primary" className="mb-2" />
            <div className="text-muted small">
              <p className="mb-1">
                Running federated learning simulation with {cfg.num_of_clients} clients over{' '}
                {cfg.num_of_rounds} rounds...
              </p>
              <p className="mb-0">
                Status updates every 2 seconds. Results will appear automatically when complete.
              </p>
            </div>
          </Card.Body>
        </Card>
      )}

      {displayStatus === 'completed' && (
        <Alert variant="success" className="mb-3">
          <strong>Simulation completed successfully!</strong> View results in the tabs below.
        </Alert>
      )}

      {displayStatus === 'failed' && (
        <Alert variant="danger" className="mb-3">
          <strong>Simulation failed.</strong> Check the error logs for details.
        </Alert>
      )}

      <Tabs defaultActiveKey="insights" className="mt-4 flex-nowrap overflow-auto">
        <Tab eventKey="insights" title="Insights">
          <InsightsTab details={details} csvData={csvData} status={displayStatus} />
        </Tab>

        <Tab eventKey="plots" title="Plots">
          <PlotsTab simulation={{ ...details, id: simulationId }} />
        </Tab>

        <Tab eventKey="metrics" title="Metrics">
          <MetricsTab
            csvData={csvData}
            csvFiles={csvFiles}
            config={details.config}
            simulationId={simulationId}
          />
        </Tab>

        <Tab eventKey="config" title="Config">
          <Card className="mt-3">
            <Card.Body>
              <ConfigTab config={details.config} />
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </PageContainer>
  );
}
