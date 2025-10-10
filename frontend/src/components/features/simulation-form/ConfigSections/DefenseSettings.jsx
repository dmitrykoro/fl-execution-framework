import { NumberField } from '../FormFields/NumberField';
import { SwitchField } from '../FormFields/SwitchField';

export function DefenseSettings({ config, onChange }) {
  const needsTrustParams = config.aggregation_strategy_keyword === 'trust';
  const needsPidParams = ['pid', 'pid_scaled', 'pid_standardized'].includes(
    config.aggregation_strategy_keyword
  );
  const needsKrumParams = ['multi-krum', 'krum', 'multi-krum-based'].includes(
    config.aggregation_strategy_keyword
  );
  const needsTrimmedMeanParams = config.aggregation_strategy_keyword === 'trimmed_mean';

  const hasDefenseParams =
    needsTrustParams || needsPidParams || needsKrumParams || needsTrimmedMeanParams;

  if (!hasDefenseParams) {
    return (
      <div className="text-muted fst-italic">
        No strategy-specific parameters for {config.aggregation_strategy_keyword}
      </div>
    );
  }

  return (
    <>
      {needsTrustParams && (
        <>
          <SwitchField
            name="remove_clients"
            label="Remove Malicious Clients"
            checked={config.remove_clients === 'true'}
            onChange={e =>
              onChange({
                target: { name: 'remove_clients', value: e.target.checked ? 'true' : 'false' },
              })
            }
            tooltip="Enable client removal based on trust scores"
          />

          <NumberField
            name="begin_removing_from_round"
            label="Begin Removing From Round"
            value={config.begin_removing_from_round}
            onChange={onChange}
            min={1}
            tooltip="Round number when trust-based client filtering starts. Earlier = more aggressive filtering."
          />

          <NumberField
            name="trust_threshold"
            label="Trust Threshold"
            value={config.trust_threshold}
            onChange={onChange}
            step={0.01}
            min={0}
            max={1}
            tooltip="Minimum trust score (0-1) for client inclusion. Lower = more permissive, higher = stricter filtering."
          />

          <NumberField
            name="beta_value"
            label="Beta Value"
            value={config.beta_value}
            onChange={onChange}
            step={0.01}
            min={0}
            max={1}
            tooltip="Exponential moving average decay factor for trust score updates (0-1). Higher values (closer to 1) give more weight to historical behavior; lower values react faster to recent changes."
          />

          <NumberField
            name="num_of_clusters"
            label="Number of Clusters"
            value={config.num_of_clusters}
            onChange={onChange}
            min={1}
            tooltip="Number of client clusters for trust grouping. More clusters = finer-grained trust analysis."
          />
        </>
      )}

      {needsPidParams && (
        <>
          <NumberField
            name="num_std_dev"
            label="Number of Std Deviations"
            value={config.num_std_dev}
            onChange={onChange}
            step={0.1}
            min={0}
            tooltip="Threshold for outlier detection. Updates beyond this many standard deviations are filtered."
          />

          <NumberField
            name="Kp"
            label="Kp (Proportional Gain)"
            value={config.Kp}
            onChange={onChange}
            step={0.01}
            tooltip="PID controller proportional term. Controls reaction to current error. Higher = more aggressive correction."
          />

          <NumberField
            name="Ki"
            label="Ki (Integral Gain)"
            value={config.Ki}
            onChange={onChange}
            step={0.01}
            tooltip="PID controller integral term. Eliminates steady-state error by accumulating past errors."
          />

          <NumberField
            name="Kd"
            label="Kd (Derivative Gain)"
            value={config.Kd}
            onChange={onChange}
            step={0.01}
            tooltip="PID controller derivative term. Predicts future error based on rate of change. Reduces overshoot."
          />
        </>
      )}

      {needsKrumParams && (
        <NumberField
          name="num_krum_selections"
          label="Krum Selections"
          value={config.num_krum_selections}
          onChange={onChange}
          min={1}
          tooltip="Number of closest clients to aggregate. Lower = more Byzantine robustness but less data diversity."
        />
      )}

      {needsTrimmedMeanParams && (
        <NumberField
          name="trim_ratio"
          label="Trim Ratio"
          value={config.trim_ratio}
          onChange={onChange}
          step={0.01}
          min={0}
          max={0.5}
          tooltip="Fraction of extreme values to remove from both ends (0-0.5). Higher = more aggressive outlier filtering."
        />
      )}
    </>
  );
}
