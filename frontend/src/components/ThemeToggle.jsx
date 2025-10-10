import { Nav } from 'react-bootstrap';
import { useTheme } from '../contexts/ThemeContext';
import { MaterialIcon } from './common/Icon/MaterialIcon';
import './ThemeToggle.css';

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Nav.Link
      as="button"
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      <span className="theme-toggle-icon">
        {theme === 'light' ? (
          <MaterialIcon name="dark_mode" size={20} />
        ) : (
          <MaterialIcon name="light_mode" size={20} />
        )}
      </span>
      <span className="theme-toggle-text">{theme === 'light' ? 'Dark Mode' : 'Light Mode'}</span>
    </Nav.Link>
  );
}

export default ThemeToggle;
