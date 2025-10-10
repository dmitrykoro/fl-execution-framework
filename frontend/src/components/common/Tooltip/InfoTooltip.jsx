import { OverlayTrigger, Tooltip } from 'react-bootstrap';

export function InfoTooltip({ text, placement = 'right', children }) {
  if (!text) {
    return children;
  }

  return (
    <OverlayTrigger placement={placement} overlay={<Tooltip>{text}</Tooltip>}>
      <span style={{ cursor: 'help', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
        {children}
        <span style={{ fontSize: '0.9rem' }}>ℹ️</span>
      </span>
    </OverlayTrigger>
  );
}
