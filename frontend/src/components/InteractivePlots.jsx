import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { Card, Form, ButtonGroup, Button } from 'react-bootstrap';
import { useTheme } from '../contexts/ThemeContext';
import RoundMetricsPlot from './RoundMetricsPlot';

const COLORS = [
  '#8884d8',
  '#82ca9d',
  '#ffc658',
  '#ff7c7c',
  '#8dd1e1',
  '#d084d0',
  '#a4de6c',
  '#ffab91',
  '#ce93d8',
  '#90caf9',
];

const MALICIOUS_COLOR = '#ff4444';

// Human-readable metric labels
const METRIC_LABELS = {
  removal_criterion_history: 'Removal Criterion',
  absolute_distance_history: 'Model Distance',
  loss_history: 'Training Loss',
  accuracy_history: 'Training Accuracy',
  trust_score_history: 'Trust Score',
  participation_history: 'Participation',
  aggregation_participation_history: 'Aggregation Participation',
};

export default function InteractivePlots({ simulation }) {
  const { theme } = useTheme();
  const [plotData, setPlotData] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('');
  const [visibleClients, setVisibleClients] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Only fetch plot data if simulation is completed
    if (simulation.status !== 'completed') {
      setLoading(false);
      return;
    }

    const fetchPlotData = async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/api/simulations/${simulation.id}/plot-data`
        );
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
  }, [simulation.id, simulation.status]);

  // Auto-select first available metric if current selection is not available
  useEffect(() => {
    if (!plotData) return;

    const allMetrics =
      plotData.per_client_metrics.length > 0
        ? Object.keys(plotData.per_client_metrics[0].metrics)
        : [];

    const metrics = allMetrics.filter(metricName =>
      plotData.per_client_metrics.some(client =>
        client.metrics[metricName]?.some(value => value !== null && value !== undefined)
      )
    );

    if (metrics.length > 0 && (!selectedMetric || !metrics.includes(selectedMetric))) {
      setSelectedMetric(metrics[0]);
    }
  }, [plotData, selectedMetric]);

  // Responsive chart height
  const [chartHeight, setChartHeight] = useState(window.innerWidth < 768 ? 350 : 500);

  useEffect(() => {
    const updateHeight = () => {
      const isMobile = window.innerWidth < 768;
      setChartHeight(isMobile ? 350 : 500);
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  if (loading) return <div className="text-center p-4">Loading interactive plots...</div>;
  if (!plotData) {
    if (simulation.status === 'running') {
      return (
        <div className="text-center p-4">
          ⏳ Plots will be available when the simulation completes...
        </div>
      );
    }
    return <div className="text-center p-4">No plot data available</div>;
  }

  // Get all metric names
  const allMetrics =
    plotData.per_client_metrics.length > 0
      ? Object.keys(plotData.per_client_metrics[0].metrics)
      : [];

  // Filter to only show metrics that have non-null data
  const metrics = allMetrics.filter(metricName =>
    plotData.per_client_metrics.some(client =>
      client.metrics[metricName]?.some(value => value !== null && value !== undefined)
    )
  );

  // Check if all metrics are null (empty data)
  if (metrics.length === 0) {
    // Show round-level metrics instead
    return <RoundMetricsPlot plotData={plotData} />;
  }

  // Transform data for Recharts
  const chartData = plotData.rounds.map((round, idx) => {
    const point = { round, name: `Round ${round}` }; // Add 'name' for Brush aria-label

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

  const toggleClient = clientKey => {
    setVisibleClients(prev => ({
      ...prev,
      [clientKey]: !prev[clientKey],
    }));
  };

  // Theme-aware chart colors
  const chartColors =
    theme === 'dark'
      ? {
          grid: '#444',
          axis: '#999',
          text: '#ccc',
          brush: '#666',
        }
      : {
          grid: '#e0e0e0',
          axis: '#666',
          text: '#333',
          brush: '#8884d8',
        };

  return (
    <div className="mb-4">
      <h5 className="mb-3">Interactive Plots</h5>
      <div>
        {/* Metric Selector */}
        <Form.Group className="mb-3" style={{ maxWidth: '60%', margin: '0 auto' }}>
          <Form.Label>Select Metric:</Form.Label>
          <Form.Select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}>
            {metrics.map(metric => (
              <option key={metric} value={metric}>
                {METRIC_LABELS[metric] || metric}
              </option>
            ))}
          </Form.Select>
        </Form.Group>

        {/* Legend Controls */}
        <div className="mb-3 text-center">
          <small className="text-muted d-block mb-2">Toggle clients:</small>
          <div className="d-flex flex-wrap gap-2 justify-content-center">
            {plotData.per_client_metrics.map((client, idx) => {
              const clientKey = `client_${client.client_id}`;
              const color = client.is_malicious ? MALICIOUS_COLOR : COLORS[idx % COLORS.length];
              return (
                <Button
                  key={clientKey}
                  size="sm"
                  variant={visibleClients[clientKey] ? 'primary' : 'outline-secondary'}
                  onClick={() => toggleClient(clientKey)}
                  className="py-2 px-3"
                  style={{
                    backgroundColor: visibleClients[clientKey] ? color : 'transparent',
                    borderColor: color,
                    color: visibleClients[clientKey] ? 'white' : color,
                  }}
                >
                  Client {client.client_id}
                  {client.is_malicious ? ' (M)' : ''}
                </Button>
              );
            })}
            {plotData.removal_threshold_history &&
              selectedMetric === 'removal_criterion_history' && (
                <Button
                  size="sm"
                  variant={visibleClients.threshold ? 'danger' : 'outline-danger'}
                  onClick={() => toggleClient('threshold')}
                  className="py-2 px-3"
                >
                  Threshold
                </Button>
              )}
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={chartHeight}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 80 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis
              dataKey="round"
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text, fontSize: chartHeight < 400 ? 10 : 12 }}
              label={{
                value: 'Round #',
                position: 'insideBottom',
                offset: -10,
                fill: chartColors.text,
                fontSize: chartHeight < 400 ? 10 : 12,
              }}
              height={60}
            />
            <YAxis
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text, fontSize: chartHeight < 400 ? 10 : 12 }}
              label={{
                value: METRIC_LABELS[selectedMetric] || selectedMetric,
                angle: -90,
                position: 'insideLeft',
                fill: chartColors.text,
                fontSize: chartHeight < 400 ? 10 : 12,
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#2b2b2b' : '#fff',
                border: `1px solid ${chartColors.grid}`,
                color: chartColors.text,
              }}
            />
            <Legend wrapperStyle={{ color: chartColors.text }} />
            <Brush
              dataKey="round"
              height={30}
              stroke={chartColors.brush}
              fill={theme === 'dark' ? '#1a1a1a' : '#f5f5f5'}
              y={chartHeight - 70}
            />

            {plotData.per_client_metrics.map((client, idx) => {
              const clientKey = `client_${client.client_id}`;
              const color = client.is_malicious ? MALICIOUS_COLOR : COLORS[idx % COLORS.length];
              return (
                visibleClients[clientKey] && (
                  <Line
                    key={clientKey}
                    type="monotone"
                    dataKey={clientKey}
                    stroke={color}
                    strokeWidth={2}
                    strokeDasharray={client.is_malicious ? '5 5' : undefined}
                    dot={{ r: 3 }}
                    name={`Client ${client.client_id}${client.is_malicious ? ' (Malicious)' : ''}`}
                  />
                )
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
            📊 Zoom: Drag on brush below chart | 👆 Toggle: Tap client buttons to show/hide lines |
            ℹ️ Tap chart: View exact values
          </small>
        </div>
      </div>
    </div>
  );
}
