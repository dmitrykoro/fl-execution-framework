import { Form } from 'react-bootstrap';
import { InfoTooltip } from '@components/common/Tooltip/InfoTooltip';

export function NumberField({
  name,
  label,
  value,
  onChange,
  tooltip,
  step,
  min,
  max,
  required,
  className = '',
  ...props
}) {
  const labelComponent = tooltip ? (
    <InfoTooltip text={tooltip}>
      <Form.Label>{label}</Form.Label>
    </InfoTooltip>
  ) : (
    <Form.Label>{label}</Form.Label>
  );

  return (
    <Form.Group className={`mb-3 ${className}`}>
      {labelComponent}
      <Form.Control
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        step={step}
        min={min}
        max={max}
        required={required}
        {...props}
      />
    </Form.Group>
  );
}
