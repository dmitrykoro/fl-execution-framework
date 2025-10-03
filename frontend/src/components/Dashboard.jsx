import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Spinner, Alert, Button, Badge, Tooltip, OverlayTrigger, Form } from 'react-bootstrap';
import useApi from '../hooks/useApi';
import { getSimulations, getSimulationStatus } from '../api';

function Dashboard() {
  const { data: simulations, loading, error } = useApi(getSimulations);
  const [statuses, setStatuses] = useState({});
  const [selectedSims, setSelectedSims] = useState([]);
  const navigate = useNavigate();

  const getRelativeTime = (timestamp) => {
    if (!timestamp) return '';
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now - then;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return then.toLocaleDateString();
  };

  const parseErrorMessage = (errorText) => {
    if (!errorText) return 'Unknown error occurred';

    // Check for validation errors (missing required fields)
    if (errorText.includes("is a required property")) {
      const match = errorText.match(/'([^']+)' is a required property/);
      const field = match ? match[1] : 'field';
      return `Configuration Error: Missing required field '${field}'`;
    }

    // Check for config loading errors
    if (errorText.includes("Error while loading config")) {
      return 'Configuration Error: Invalid or incomplete configuration';
    }

    // Check for enum validation errors
    if (errorText.includes("is not one of")) {
      const match = errorText.match(/'([^']+)' is not one of/);
      const value = match ? match[1] : 'value';
      return `Configuration Error: Invalid value '${value}'`;
    }

    // Default: take first line of error if multiline
    const lines = errorText.split('\n');
    const firstMeaningfulLine = lines.find(line =>
      line.includes('ERROR') || line.includes('Error') || line.trim().length > 0
    );

    if (firstMeaningfulLine) {
      // Clean up the line (remove ERROR: prefix, trim)
      return firstMeaningfulLine.replace(/^ERROR:root:/i, '').replace(/^ERROR:/i, '').trim();
    }

    return 'An error occurred during simulation';
  };

  useEffect(() => {
    if (!simulations) return;

    const fetchStatuses = async () => {
      const newStatuses = {};
      for (const sim of simulations) {
        try {
          const response = await getSimulationStatus(sim.simulation_id);
          newStatuses[sim.simulation_id] = response.data;
        } catch {
          newStatuses[sim.simulation_id] = { status: 'unknown' };
        }
      }
      setStatuses(newStatuses);
    };

    fetchStatuses();
    const interval = setInterval(fetchStatuses, 5000);
    return () => clearInterval(interval);
  }, [simulations]);

  const handleCheckboxChange = (simId) => {
    setSelectedSims(prev =>
      prev.includes(simId) ? prev.filter(id => id !== simId) : [...prev, simId]
    );
  };

  const handleCompare = () => {
    if (selectedSims.length < 2) {
      alert('Please select at least 2 simulations to compare');
      return;
    }
    navigate(`/compare?ids=${selectedSims.join(',')}`);
  };

  const getStatusBadge = (statusData) => {
    if (!statusData) return <Badge bg="secondary">pending</Badge>;

    const { status, error: errorMsg } = statusData;
    const variants = {
      pending: 'secondary',
      running: 'primary',
      completed: 'success',
      failed: 'danger',
      unknown: 'warning'
    };

    const badge = <Badge bg={variants[status] || 'secondary'}>{status || 'pending'}</Badge>;

    if (status === 'failed' && errorMsg) {
      return (
        <OverlayTrigger
          placement="left"
          overlay={
            <Tooltip id={`tooltip-error`}>
              <div style={{ textAlign: 'left', whiteSpace: 'pre-wrap', maxWidth: '400px' }}>
                {errorMsg}
              </div>
            </Tooltip>
          }
        >
          <span style={{ cursor: 'pointer' }}>{badge}</span>
        </OverlayTrigger>
      );
    }

    return badge;
  };

  if (loading) {
    return (
      <div>
        <h1>Simulation Dashboard</h1>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>Simulation Dashboard</h1>
        <Alert variant="danger">{error}</Alert>
      </div>
    );
  }

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1>Simulation Dashboard</h1>
        <div className="d-flex gap-2">
          {selectedSims.length > 0 && (
            <Button variant="info" onClick={handleCompare}>
              📊 Compare Selected ({selectedSims.length})
            </Button>
          )}
          <Link to="/simulations/new">
            <Button variant="primary">+ New Simulation</Button>
          </Link>
        </div>
      </div>
      <Row xs={1} md={2} lg={3} className="g-4">
        {simulations && simulations.length > 0 ? (
          simulations.map(sim => {
            const statusData = statuses[sim.simulation_id];
            const isFailed = statusData?.status === 'failed';

            return (
              <Col key={sim.simulation_id}>
                <Card className={isFailed ? 'border-danger' : selectedSims.includes(sim.simulation_id) ? 'border-primary' : ''}>
                  <Card.Body>
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <div className="d-flex align-items-start gap-2">
                        <Form.Check
                          type="checkbox"
                          checked={selectedSims.includes(sim.simulation_id)}
                          onChange={() => handleCheckboxChange(sim.simulation_id)}
                          style={{ marginTop: '0.25rem' }}
                        />
                        <Card.Title className="mb-0">{sim.strategy_name}</Card.Title>
                      </div>
                      {getStatusBadge(statusData)}
                    </div>
                    <Card.Subtitle className="mb-2 text-muted">
                      {sim.simulation_id}
                      {sim.created_at && (
                        <span className="ms-2 small">• {getRelativeTime(sim.created_at)}</span>
                      )}
                    </Card.Subtitle>
                    {isFailed && statusData.error && (
                      <Alert variant="danger" className="mb-2 py-2">
                        {parseErrorMessage(statusData.error)}
                      </Alert>
                    )}
                    <Card.Text>
                      Rounds: {sim.num_of_rounds} | Clients: {sim.num_of_clients}
                    </Card.Text>
                    <Link to={`/simulations/${sim.simulation_id}`}>View Details</Link>
                  </Card.Body>
                </Card>
              </Col>
            );
          })
        ) : (
          <Col>
            <p>No simulations found.</p>
          </Col>
        )}
      </Row>
    </div>
  );
}

export default Dashboard;
