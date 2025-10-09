import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { Nav } from 'react-bootstrap';
import { useTheme } from '../contexts/ThemeContext';
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
        {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
      </span>
      <span className="theme-toggle-text">
        {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
      </span>
    </Nav.Link>
  );
}

export default ThemeToggle;
