import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, Form } from 'react-bootstrap';
import { useTheme } from '../contexts/ThemeContext';

const METRIC_LABELS = {
  aggregated_loss_history: 'Aggregated Loss',
  average_accuracy_history: 'Average Accuracy',
  score_calculation_time_nanos_history: 'Score Calculation Time (ns)',
};

export default function RoundMetricsPlot({ plotData }) {
  const { theme } = useTheme();
  const [selectedMetric, setSelectedMetric] = useState('');

  // Get available round metrics
  const availableMetrics = plotData.round_metrics
    ? Object.keys(plotData.round_metrics).filter(
        key => plotData.round_metrics[key] && plotData.round_metrics[key].length > 0
      )
    : [];

  // Set default metric if not set
  if (!selectedMetric && availableMetrics.length > 0) {
    setSelectedMetric(availableMetrics[0]);
  }

  // Transform data for Recharts
  const chartData = plotData.rounds.map((round, idx) => {
    const point = { round };
    if (plotData.round_metrics && plotData.round_metrics[selectedMetric]) {
      point.value = plotData.round_metrics[selectedMetric][idx];
    }
    return point;
  });

  // Theme-aware chart colors
  const chartColors =
    theme === 'dark'
      ? {
          grid: '#444',
          axis: '#999',
          text: '#ccc',
          line: '#82ca9d',
        }
      : {
          grid: '#e0e0e0',
          axis: '#666',
          text: '#333',
          line: '#6750A4',
        };

  if (availableMetrics.length === 0) {
    return (
      <div className="text-center p-4">
        <p className="text-muted">No round-level metrics available for this simulation.</p>
      </div>
    );
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5>Round-Level Convergence Metrics</h5>
      </Card.Header>
      <Card.Body>
        {/* Metric Selector */}
        <Form.Group className="mb-3">
          <Form.Label>Select Metric:</Form.Label>
          <Form.Select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}>
            {availableMetrics.map(metric => (
              <option key={metric} value={metric}>
                {METRIC_LABELS[metric] || metric}
              </option>
            ))}
          </Form.Select>
        </Form.Group>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis
              dataKey="round"
              label={{ value: 'Round', position: 'insideBottom', offset: -5 }}
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text }}
            />
            <YAxis
              label={{
                value: METRIC_LABELS[selectedMetric] || selectedMetric,
                angle: -90,
                position: 'insideLeft',
              }}
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#2b2930' : '#fff',
                border: `1px solid ${chartColors.grid}`,
                color: chartColors.text,
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              name={METRIC_LABELS[selectedMetric] || selectedMetric}
              stroke={chartColors.line}
              strokeWidth={2}
              dot={{ fill: chartColors.line, r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-3">
          <small className="text-muted">
            This chart shows aggregated metrics across all participating clients per round.
            {selectedMetric === 'aggregated_loss_history' &&
              ' Lower loss indicates better model convergence.'}
            {selectedMetric === 'average_accuracy_history' &&
              ' Higher accuracy indicates better model performance.'}
          </small>
        </div>
      </Card.Body>
    </Card>
  );
}
