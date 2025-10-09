import { Button, OverlayTrigger, Tooltip } from 'react-bootstrap';

/**
 * Reusable outline button component with consistent styling and optional tooltip
 *
 * @param {string} variant - Bootstrap variant (outline-secondary, outline-primary, etc.)
 * @param {string} size - Button size (sm, md, lg)
 * @param {function} onClick - Click handler
 * @param {boolean} disabled - Whether button is disabled
 * @param {string} tooltip - Optional tooltip text to show on hover
 * @param {string} tooltipPlacement - Tooltip position (top, bottom, left, right)
 * @param {string} className - Additional CSS classes
 * @param {object} style - Additional inline styles
 * @param {string} as - Render as different element (e.g., 'label')
 * @param {string} htmlFor - For label elements
 * @param {ReactNode} children - Button content
 */
function OutlineButton({
  variant = 'outline-secondary',
  size = 'sm',
  onClick,
  disabled = false,
  tooltip,
  tooltipPlacement = 'bottom',
  className = '',
  style = {},
  as,
  htmlFor,
  children,
  ...otherProps
}) {
  // Default styles for consistency
  const defaultStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...style,
  };

  const button = (
    <Button
      variant={variant}
      size={size}
      onClick={onClick}
      disabled={disabled}
      className={className}
      style={defaultStyle}
      as={as}
      htmlFor={htmlFor}
      {...otherProps}
    >
      {children}
    </Button>
  );

  // Wrap with tooltip if provided
  if (tooltip) {
    return (
      <OverlayTrigger placement={tooltipPlacement} overlay={<Tooltip>{tooltip}</Tooltip>}>
        {button}
      </OverlayTrigger>
    );
  }

  return button;
}

export default OutlineButton;
