/**
 * Material Symbols icon wrapper component
 *
 * @see https://fonts.google.com/icons
 * @see https://developers.google.com/fonts/docs/material_symbols
 */
export function MaterialIcon({
  name,
  size = 24,
  fill = 0, // 0 = outlined, 1 = filled
  weight = 400, // 100-700 (thin to bold)
  grade = 0, // -25 to 200 (thinner to thicker)
  className = '',
  style = {},
  ...props
}) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{
        fontSize: `${size}px`,
        fontVariationSettings: `'FILL' ${fill}, 'wght' ${weight}, 'GRAD' ${grade}, 'opsz' ${size}`,
        display: 'inline-flex',
        alignItems: 'center',
        ...style,
      }}
      {...props}
    >
      {name}
    </span>
  );
}
