import { Form } from 'react-bootstrap';
import { InfoTooltip } from '@components/common/Tooltip/InfoTooltip';

export function SwitchField({ name, label, checked, onChange, tooltip, className = '', ...props }) {
  const labelComponent = tooltip ? (
    <InfoTooltip text={tooltip}>
      <span>{label}</span>
    </InfoTooltip>
  ) : (
    <span>{label}</span>
  );

  return (
    <Form.Group className={`mb-3 ${className}`}>
      <Form.Check
        type="switch"
        id={`switch-${name}`}
        name={name}
        checked={checked}
        onChange={onChange}
        label={labelComponent}
        {...props}
      />
    </Form.Group>
  );
}
