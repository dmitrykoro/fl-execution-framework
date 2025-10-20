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
import { Form, ButtonGroup, Button, Card } from 'react-bootstrap';
import { useTheme } from '@contexts/ThemeContext';

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
  '#ff8a65',
  '#81c784',
];

const METRIC_LABELS = {
  average_loss_history: 'Average Loss',
  average_accuracy_history: 'Average Accuracy',
  removal_accuracy_history: 'Removal Accuracy',
  removal_precision_history: 'Removal Precision',
  removal_recall_history: 'Removal Recall',
};

function generateStrategyLabel(strategyConfig, strategyNumber) {
  const parts = [];
  parts.push(strategyConfig.aggregation_strategy_keyword || 'fedavg');

  if (strategyConfig.num_krum_selections !== undefined) {
    parts.push(`k=${strategyConfig.num_krum_selections}`);
  }
  if (strategyConfig.num_of_malicious_clients !== undefined) {
    parts.push(`mal=${strategyConfig.num_of_malicious_clients}`);
  }
  if (strategyConfig.remove_clients === 'true') {
    parts.push('rm=T');
  }

  return `S${strategyNumber}: ${parts.join(' ')}`;
}

export function StrategyComparisonPlot({ allPlotData, strategyConfigs }) {
  const { theme } = useTheme();
  const [selectedMetric, setSelectedMetric] = useState('average_accuracy_history');
  const [visibleStrategies, setVisibleStrategies] = useState({});
  const [groupFilter, setGroupFilter] = useState('all');

  useEffect(() => {
    if (allPlotData && allPlotData.length > 0) {
      const visibility = {};
      allPlotData.forEach(strategy => {
        visibility[`strategy_${strategy.strategy_number}`] = true;
      });
      setVisibleStrategies(visibility);
    }
  }, [allPlotData]);

  if (!allPlotData || allPlotData.length === 0) {
    return <div className="text-center p-4">No comparison data available</div>;
  }

  const firstStrategy = allPlotData[0].data;
  const availableMetrics = firstStrategy.round_metrics
    ? Object.keys(METRIC_LABELS).filter(metric => firstStrategy.round_metrics[metric] !== undefined)
    : [];

  if (availableMetrics.length === 0) {
    return <div className="text-center p-4">No round metrics available for comparison</div>;
  }

  const strategyGroups = allPlotData.reduce((groups, strategy) => {
    const config = strategyConfigs[strategy.strategy_number];
    const key = config?.aggregation_strategy_keyword || 'fedavg';
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(strategy.strategy_number);
    return groups;
  }, {});

  const rounds = firstStrategy.rounds || [];
  const chartData = rounds.map((round, idx) => {
    const point = { round, name: `Round ${round}` };

    allPlotData.forEach(strategy => {
      const metricValues = strategy.data.round_metrics?.[selectedMetric];
      if (metricValues && metricValues[idx] !== undefined) {
        point[`strategy_${strategy.strategy_number}`] = metricValues[idx];
      }
    });

    return point;
  });

  const toggleStrategy = strategyKey => {
    setVisibleStrategies(prev => ({
      ...prev,
      [strategyKey]: !prev[strategyKey],
    }));
  };

  const handleGroupFilter = group => {
    setGroupFilter(group);
    if (group === 'all') {
      const newVisibility = {};
      allPlotData.forEach(strategy => {
        newVisibility[`strategy_${strategy.strategy_number}`] = true;
      });
      setVisibleStrategies(newVisibility);
    } else {
      const newVisibility = {};
      allPlotData.forEach(strategy => {
        const config = strategyConfigs[strategy.strategy_number];
        const isInGroup = (config?.aggregation_strategy_keyword || 'fedavg') === group;
        newVisibility[`strategy_${strategy.strategy_number}`] = isInGroup;
      });
      setVisibleStrategies(newVisibility);
    }
  };

  const chartColors =
    theme === 'dark'
      ? {
          grid: '#444',
          axis: '#999',
          text: '#ccc',
        }
      : {
          grid: '#e0e0e0',
          axis: '#666',
          text: '#333',
        };

  const chartHeight = window.innerWidth < 768 ? 350 : 500;

  return (
    <Card className="mb-4">
      <Card.Body>
        <h5 className="mb-3">📊 Strategy Comparison</h5>

        {/* Metric Selector */}
        <Form.Group className="mb-3" style={{ maxWidth: '60%', margin: '0 auto' }}>
          <Form.Label>Select Metric:</Form.Label>
          <Form.Select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}>
            {availableMetrics.map(metric => (
              <option key={metric} value={metric}>
                {METRIC_LABELS[metric] || metric}
              </option>
            ))}
          </Form.Select>
        </Form.Group>

        {/* Group Filter */}
        <div className="mb-3 text-center">
          <small className="text-muted d-block mb-2">Filter by strategy type:</small>
          <ButtonGroup size="sm">
            <Button
              variant={groupFilter === 'all' ? 'primary' : 'outline-primary'}
              onClick={() => handleGroupFilter('all')}
            >
              All Strategies ({allPlotData.length})
            </Button>
            {Object.keys(strategyGroups).map(group => (
              <Button
                key={group}
                variant={groupFilter === group ? 'primary' : 'outline-primary'}
                onClick={() => handleGroupFilter(group)}
              >
                {group} ({strategyGroups[group].length})
              </Button>
            ))}
          </ButtonGroup>
        </div>

        {/* Strategy Toggle Controls */}
        <div className="mb-3 text-center">
          <small className="text-muted d-block mb-2">Toggle individual strategies:</small>
          <div className="d-flex flex-wrap gap-2 justify-content-center">
            {allPlotData.map((strategy, idx) => {
              const strategyKey = `strategy_${strategy.strategy_number}`;
              const config = strategyConfigs[strategy.strategy_number];
              const color = COLORS[idx % COLORS.length];
              const label = generateStrategyLabel(config, strategy.strategy_number);

              return (
                <Button
                  key={strategyKey}
                  size="sm"
                  variant={visibleStrategies[strategyKey] ? 'primary' : 'outline-secondary'}
                  onClick={() => toggleStrategy(strategyKey)}
                  className="py-1 px-2"
                  style={{
                    backgroundColor: visibleStrategies[strategyKey] ? color : 'transparent',
                    borderColor: color,
                    color: visibleStrategies[strategyKey] ? 'white' : color,
                    fontSize: '0.75rem',
                  }}
                >
                  {label}
                </Button>
              );
            })}
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={chartHeight}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 80 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis
              dataKey="round"
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text, fontSize: 10 }}
              label={{
                value: 'Round #',
                position: 'insideBottom',
                offset: -10,
                fill: chartColors.text,
              }}
              height={60}
            />
            <YAxis
              stroke={chartColors.axis}
              tick={{ fill: chartColors.text, fontSize: 10 }}
              label={{
                value: METRIC_LABELS[selectedMetric] || selectedMetric,
                angle: -90,
                position: 'insideLeft',
                fill: chartColors.text,
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#2b2b2b' : '#fff',
                border: `1px solid ${chartColors.grid}`,
                color: chartColors.text,
                maxHeight: '300px',
                overflow: 'auto',
              }}
            />
            <Legend wrapperStyle={{ color: chartColors.text, fontSize: '0.75rem' }} />
            <Brush
              dataKey="round"
              height={30}
              stroke={chartColors.axis}
              fill={theme === 'dark' ? '#1a1a1a' : '#f5f5f5'}
              y={chartHeight - 70}
            />

            {allPlotData.map((strategy, idx) => {
              const strategyKey = `strategy_${strategy.strategy_number}`;
              const config = strategyConfigs[strategy.strategy_number];
              const color = COLORS[idx % COLORS.length];
              const label = generateStrategyLabel(config, strategy.strategy_number);

              return (
                visibleStrategies[strategyKey] && (
                  <Line
                    key={strategyKey}
                    type="monotone"
                    dataKey={strategyKey}
                    stroke={color}
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    name={label}
                  />
                )
              );
            })}
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-3 text-muted">
          <small>
            📊 Zoom: Drag on brush | 👆 Filter: Use group buttons | 🔘 Toggle: Click strategy
            buttons
          </small>
        </div>
      </Card.Body>
    </Card>
  );
}
