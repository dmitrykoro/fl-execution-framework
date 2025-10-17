// Quick pattern presets for experiment queue strategy variations

export const QUICK_PATTERNS = {
  sweepMalicious: {
    id: 'sweepMalicious',
    name: 'Sweep Malicious Clients (0-5)',
    description: 'Test defense robustness by increasing malicious clients from 0 to 5',
    generate: baseStrategy => {
      return Array.from({ length: 6 }, (_, i) => ({
        name: `${baseStrategy || 'fedavg'}_mal_${i}`,
        aggregation_strategy_keyword: baseStrategy || 'fedavg',
        num_of_malicious_clients: i,
        remove_clients: 'true',
      }));
    },
  },

  compareStrategies: {
    id: 'compareStrategies',
    name: 'Compare Aggregation Strategies',
    description: 'Compare FedAvg, Krum, Multi-Krum, and Trust strategies with no attacks',
    generate: () => {
      return [
        {
          name: 'fedavg_baseline',
          aggregation_strategy_keyword: 'fedavg',
          num_of_malicious_clients: 0,
          remove_clients: 'false',
        },
        {
          name: 'krum_baseline',
          aggregation_strategy_keyword: 'krum',
          num_of_malicious_clients: 0,
          remove_clients: 'false',
        },
        {
          name: 'multi_krum_baseline',
          aggregation_strategy_keyword: 'multi-krum',
          num_of_malicious_clients: 0,
          num_krum_selections: 4,
          remove_clients: 'false',
        },
        {
          name: 'trust_baseline',
          aggregation_strategy_keyword: 'trust',
          num_of_malicious_clients: 0,
          remove_clients: 'false',
        },
      ];
    },
  },

  sweepKrumSelections: {
    id: 'sweepKrumSelections',
    name: 'Sweep Krum Selections (1-5)',
    description: 'Test multi-krum with different selection thresholds',
    generate: () => {
      return [1, 2, 3, 4, 5].map(k => ({
        name: `multi_krum_k${k}`,
        aggregation_strategy_keyword: 'multi-krum',
        num_of_malicious_clients: 0,
        num_krum_selections: k,
        remove_clients: 'false',
      }));
    },
  },

  defenseComparison: {
    id: 'defenseComparison',
    name: 'Defense Comparison (1 Malicious)',
    description: 'Compare defense strategies under attack (1 malicious client)',
    generate: () => {
      return [
        {
          name: 'fedavg_1mal',
          aggregation_strategy_keyword: 'fedavg',
          num_of_malicious_clients: 1,
          remove_clients: 'false',
        },
        {
          name: 'krum_1mal',
          aggregation_strategy_keyword: 'krum',
          num_of_malicious_clients: 1,
          remove_clients: 'true',
        },
        {
          name: 'multi_krum_1mal',
          aggregation_strategy_keyword: 'multi-krum',
          num_of_malicious_clients: 1,
          num_krum_selections: 3,
          remove_clients: 'true',
        },
        {
          name: 'trust_1mal',
          aggregation_strategy_keyword: 'trust',
          num_of_malicious_clients: 1,
          remove_clients: 'true',
        },
      ];
    },
  },
};

// Strategy parameter fields that can be overridden per strategy
export const OVERRIDABLE_FIELDS = [
  'aggregation_strategy_keyword',
  'num_of_malicious_clients',
  'num_krum_selections',
  'remove_clients',
  'begin_removing_from_round',
  'trust_threshold',
  'beta_value',
  'Kp',
  'Ki',
  'Kd',
  'attack_type',
  'gaussian_noise_std',
];

// Aggregation strategy options
export const AGGREGATION_STRATEGIES = [
  { value: 'fedavg', label: 'FedAvg' },
  { value: 'krum', label: 'Krum' },
  { value: 'multi-krum', label: 'Multi-Krum' },
  { value: 'trust', label: 'Trust' },
  { value: 'pid', label: 'PID Controller' },
];
