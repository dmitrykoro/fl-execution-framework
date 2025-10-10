import { SwitchField } from '../FormFields/SwitchField';

export function OutputSettings({ config, onChange }) {
  return (
    <>
      <SwitchField
        name="show_plots"
        label="Show Plots"
        checked={config.show_plots === 'true'}
        onChange={e =>
          onChange({ target: { name: 'show_plots', value: e.target.checked ? 'true' : 'false' } })
        }
        tooltip="Display plots during simulation (may slow down execution)"
      />

      <SwitchField
        name="save_plots"
        label="Save Plots"
        checked={config.save_plots === 'true'}
        onChange={e =>
          onChange({ target: { name: 'save_plots', value: e.target.checked ? 'true' : 'false' } })
        }
        tooltip="Save plot images to disk after simulation completes"
      />

      <SwitchField
        name="save_csv"
        label="Save CSV"
        checked={config.save_csv === 'true'}
        onChange={e =>
          onChange({ target: { name: 'save_csv', value: e.target.checked ? 'true' : 'false' } })
        }
        tooltip="Export metrics to CSV files for further analysis"
      />

      <SwitchField
        name="preserve_dataset"
        label="Preserve Dataset"
        checked={config.preserve_dataset === 'true'}
        onChange={e =>
          onChange({
            target: { name: 'preserve_dataset', value: e.target.checked ? 'true' : 'false' },
          })
        }
        tooltip="Keep partitioned dataset files after simulation (useful for reproducibility)"
      />

      <SwitchField
        name="strict_mode"
        label="Strict Mode"
        checked={config.strict_mode === 'true'}
        onChange={e =>
          onChange({ target: { name: 'strict_mode', value: e.target.checked ? 'true' : 'false' } })
        }
        tooltip="Enable strict validation of configuration parameters"
      />

      <SwitchField
        name="remove_clients"
        label="Remove Clients"
        checked={config.remove_clients === 'true'}
        onChange={e =>
          onChange({
            target: { name: 'remove_clients', value: e.target.checked ? 'true' : 'false' },
          })
        }
        tooltip="Enable client removal based on defense strategy (required for some strategies)"
      />
    </>
  );
}
