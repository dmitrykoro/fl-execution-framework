import { SelectField } from '../FormFields/SelectField';
import { NumberField } from '../FormFields/NumberField';
import { DEVICES } from '@constants/attacks';

export function TrainingSettings({ config, onChange }) {
  return (
    <>
      <NumberField
        name="batch_size"
        label="Batch Size"
        value={config.batch_size}
        onChange={onChange}
        min={1}
        tooltip="Number of samples per training batch. Larger = faster but more memory. 32 is standard."
        required
      />

      <NumberField
        name="num_of_client_epochs"
        label="Client Epochs"
        value={config.num_of_client_epochs}
        onChange={onChange}
        min={1}
        tooltip="Number of complete passes over local data each client performs before sending updates to server. More epochs = better local training but higher computation cost."
        required
      />

      <SelectField
        name="training_device"
        label="Training Device"
        value={config.training_device}
        onChange={onChange}
        options={DEVICES}
        tooltip="Hardware for training. CPU is most compatible, GPU/CUDA requires NVIDIA GPU."
      />

      <NumberField
        name="cpus_per_client"
        label="CPUs per Client"
        value={config.cpus_per_client}
        onChange={onChange}
        min={1}
        tooltip="Number of CPU cores allocated to each simulated client for parallel processing. Adjust based on available system resources."
      />

      <NumberField
        name="gpus_per_client"
        label="GPUs per Client"
        value={config.gpus_per_client}
        onChange={onChange}
        step={0.1}
        min={0}
        tooltip="Fraction of GPU allocated to each client. 0 = CPU only, 1.0 = full GPU."
      />

      <NumberField
        name="training_subset_fraction"
        label="Training Subset Fraction"
        value={config.training_subset_fraction}
        onChange={onChange}
        step={0.001}
        min={0.001}
        max={1.0}
        tooltip="Fraction of training data to use (0.001-1.0). Lower values = faster training for testing."
      />
    </>
  );
}
