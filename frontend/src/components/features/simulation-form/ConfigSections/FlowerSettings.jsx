import { NumberField } from '../FormFields/NumberField';
import { SelectField } from '../FormFields/SelectField';

export function FlowerSettings({ config, onChange }) {
  return (
    <>
      <NumberField
        name="min_fit_clients"
        label="Min Fit Clients"
        value={config.min_fit_clients}
        onChange={onChange}
        min={1}
        tooltip="Minimum number of clients that must participate in training each round."
        required
      />

      <NumberField
        name="min_evaluate_clients"
        label="Min Evaluate Clients"
        value={config.min_evaluate_clients}
        onChange={onChange}
        min={1}
        tooltip="Minimum number of clients that must participate in evaluation each round."
        required
      />

      <NumberField
        name="min_available_clients"
        label="Min Available Clients"
        value={config.min_available_clients}
        onChange={onChange}
        min={1}
        tooltip="Minimum number of clients that must be available before starting a round."
        required
      />

      <SelectField
        name="evaluate_metrics_aggregation_fn"
        label="Metrics Aggregation Function"
        value={config.evaluate_metrics_aggregation_fn}
        onChange={onChange}
        options={['weighted_average', 'average']}
        tooltip="How to aggregate evaluation metrics from clients. Weighted average considers client dataset sizes."
      />
    </>
  );
}
