import { SelectField } from '../FormFields/SelectField';
import { NumberField } from '../FormFields/NumberField';
import { ATTACKS } from '@constants/attacks';

export function AttackSettings({ config, onChange }) {
  const needsGaussianParams = config.attack_type === 'gaussian_noise';

  return (
    <>
      <NumberField
        name="num_of_malicious_clients"
        label="Number of Malicious Clients"
        value={config.num_of_malicious_clients}
        onChange={onChange}
        min={0}
        max={config.num_of_clients}
        tooltip="Number of malicious/Byzantine clients in the simulation. Set to 0 for baseline experiments."
      />

      {config.num_of_malicious_clients > 0 && (
        <>
          <SelectField
            name="attack_type"
            label="Attack Type"
            value={config.attack_type}
            onChange={onChange}
            options={ATTACKS}
            tooltip="Type of Byzantine attack. Gaussian noise adds random perturbations to model weights. Label flipping corrupts training labels to degrade model accuracy."
          />

          {needsGaussianParams && (
            <>
              <NumberField
                name="gaussian_noise_mean"
                label="Gaussian Noise Mean"
                value={config.gaussian_noise_mean}
                onChange={onChange}
                step={0.1}
                tooltip="Mean (μ) of Gaussian noise distribution added to model weights. Typically set to 0 for zero-centered noise."
              />

              <NumberField
                name="gaussian_noise_std"
                label="Gaussian Noise Std Dev"
                value={config.gaussian_noise_std}
                onChange={onChange}
                step={0.1}
                min={0}
                tooltip="Standard deviation (σ) of Gaussian noise distribution. Higher values create stronger perturbations and more aggressive attacks."
              />

              <NumberField
                name="attack_ratio"
                label="Attack Ratio"
                value={config.attack_ratio}
                onChange={onChange}
                step={0.1}
                min={0}
                max={1}
                tooltip="Fraction of model parameters to attack (0-1). 1.0 = attack all parameters."
              />
            </>
          )}
        </>
      )}
    </>
  );
}
