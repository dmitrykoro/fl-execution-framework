import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Brush
} from 'recharts';
import { Card, Form, ButtonGroup, Button } from 'react-bootstrap';

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1',
  '#d084d0', '#a4de6c', '#ffab91', '#ce93d8', '#90caf9'
];

const MALICIOUS_COLOR = '#ff4444';

export default function InteractivePlots({ simulation }) {
  const [plotData, setPlotData] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('');
  const [visibleClients, setVisibleClients] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPlotData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/simulations/${simulation.id}/plot-data`);
        if (!response.ok) throw new Error('Failed to fetch plot data');
        const data = await response.json();
        setPlotData(data);

        // Set first available metric as default
        if (data.per_client_metrics.length > 0) {
          const metrics = Object.keys(data.per_client_metrics[0].metrics);
          if (metrics.length > 0) {
            setSelectedMetric(metrics[0]);
          }
        }

        // Initialize all clients as visible
        const clientsVisibility = {};
        data.per_client_metrics.forEach(client => {
          clientsVisibility[`client_${client.client_id}`] = true;
        });
        if (data.removal_threshold_history) {
          clientsVisibility.threshold = true;
        }
        setVisibleClients(clientsVisibility);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching plot data:', error);
        setLoading(false);
      }
    };

    fetchPlotData();
  }, [simulation.id]);

  if (loading) return <div className="text-center p-4">Loading interactive plots...</div>;
  if (!plotData) return <div className="text-center p-4">No plot data available</div>;

  const metrics = plotData.per_client_metrics.length > 0
    ? Object.keys(plotData.per_client_metrics[0].metrics)
    : [];

  // Transform data for Recharts
  const chartData = plotData.rounds.map((round, idx) => {
    const point = { round };

    plotData.per_client_metrics.forEach(client => {
      const metricValues = client.metrics[selectedMetric];
      if (metricValues && metricValues[idx] !== undefined) {
        point[`client_${client.client_id}`] = metricValues[idx];
      }
    });

    if (plotData.removal_threshold_history && selectedMetric === 'removal_criterion_history') {
      point.threshold = plotData.removal_threshold_history[idx];
    }

    return point;
  });

  const toggleClient = (clientKey) => {
    setVisibleClients(prev => ({
      ...prev,
      [clientKey]: !prev[clientKey]
    }));
  };

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5>Interactive Plots</h5>
      </Card.Header>
      <Card.Body>
        {/* Metric Selector */}
        <Form.Group className="mb-3">
          <Form.Label>Select Metric:</Form.Label>
          <Form.Select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            {metrics.map(metric => (
              <option key={metric} value={metric}>{metric}</option>
            ))}
          </Form.Select>
        </Form.Group>

        {/* Legend Controls */}
        <div className="mb-3">
          <small className="text-muted">Toggle clients:</small>
          <ButtonGroup size="sm" className="ms-2 flex-wrap">
            {plotData.per_client_metrics.map((client, idx) => {
              const clientKey = `client_${client.client_id}`;
              const color = client.is_malicious ? MALICIOUS_COLOR : COLORS[idx % COLORS.length];
              return (
                <Button
                  key={clientKey}
                  variant={visibleClients[clientKey] ? 'primary' : 'outline-secondary'}
                  onClick={() => toggleClient(clientKey)}
                  style={{
                    backgroundColor: visibleClients[clientKey] ? color : 'transparent',
                    borderColor: color,
                    color: visibleClients[clientKey] ? 'white' : color
                  }}
                >
                  Client {client.client_id}{client.is_malicious ? ' (M)' : ''}
                </Button>
              );
            })}
            {plotData.removal_threshold_history && selectedMetric === 'removal_criterion_history' && (
              <Button
                variant={visibleClients.threshold ? 'danger' : 'outline-danger'}
                onClick={() => toggleClient('threshold')}
              >
                Threshold
              </Button>
            )}
          </ButtonGroup>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="round"
              label={{ value: 'Round #', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              label={{ value: selectedMetric, angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Legend />
            <Brush dataKey="round" height={30} stroke="#8884d8" />

            {plotData.per_client_metrics.map((client, idx) => {
              const clientKey = `client_${client.client_id}`;
              const color = client.is_malicious ? MALICIOUS_COLOR : COLORS[idx % COLORS.length];
              return visibleClients[clientKey] && (
                <Line
                  key={clientKey}
                  type="monotone"
                  dataKey={clientKey}
                  stroke={color}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name={`Client ${client.client_id}${client.is_malicious ? ' (Malicious)' : ''}`}
                />
              );
            })}

            {plotData.removal_threshold_history &&
             selectedMetric === 'removal_criterion_history' &&
             visibleClients.threshold && (
              <Line
                type="monotone"
                dataKey="threshold"
                stroke={MALICIOUS_COLOR}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Removal Threshold"
              />
            )}
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-3 text-muted">
          <small>
            📊 Zoom: Drag on brush below chart |
            🖱️ Toggle: Click client buttons to show/hide lines |
            ℹ️ Hover: View exact values
          </small>
        </div>
      </Card.Body>
    </Card>
  );
}
