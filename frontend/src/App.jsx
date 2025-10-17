import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { SimulationDetails } from './pages/SimulationDetails/SimulationDetails';
import { NewSimulation } from './pages/NewSimulation/NewSimulation';
import { ExperimentQueue } from './pages/ExperimentQueue/ExperimentQueue';
import { QueueStatus } from './pages/QueueStatus/QueueStatus';
import ComparisonView from './components/ComparisonView';
import ErrorBoundary from './components/ErrorBoundary';
import ThemeToggle from './components/ThemeToggle';
import { Toaster } from 'sonner';
import { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  useEffect(() => {
    const handleThemeChange = () => {
      const currentTheme = localStorage.getItem('theme') || 'light';
      setTheme(currentTheme);
    };

    // Listen for theme changes
    window.addEventListener('storage', handleThemeChange);

    // Also check for theme changes via MutationObserver on document.documentElement
    const observer = new MutationObserver(() => {
      const htmlElement = document.documentElement;
      const isDark = htmlElement.getAttribute('data-bs-theme') === 'dark';
      setTheme(isDark ? 'dark' : 'light');
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-bs-theme'],
    });

    return () => {
      window.removeEventListener('storage', handleThemeChange);
      observer.disconnect();
    };
  }, []);

  return (
    <ErrorBoundary>
      <Toaster
        position="top-right"
        theme={theme}
        toastOptions={{
          duration: 4000,
          style: {
            fontFamily: 'var(--bs-body-font-family)',
          },
        }}
      />
      <Router>
        <Navbar
          expand="md"
          className="mb-3"
          style={{ backgroundColor: 'var(--md-sys-color-surface-variant)' }}
        >
          <Container>
            <Navbar.Brand as={Link} to="/">
              FL Framework
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="navbar-nav" />
            <Navbar.Collapse id="navbar-nav">
              <Nav className="me-auto">
                <Nav.Link as={Link} to="/">
                  Dashboard
                </Nav.Link>
                <Nav.Link as={Link} to="/simulations/new">
                  New Simulation
                </Nav.Link>
                <Nav.Link as={Link} to="/experiments/queue">
                  Experiment Queue
                </Nav.Link>
              </Nav>
              <Nav>
                <ThemeToggle />
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>
        <Container className="mt-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulations/new" element={<NewSimulation />} />
            <Route path="/experiments/queue" element={<ExperimentQueue />} />
            <Route path="/queue/:simulationId" element={<QueueStatus />} />
            <Route path="/simulations/:simulationId" element={<SimulationDetails />} />
            <Route path="/compare" element={<ComparisonView />} />
          </Routes>
        </Container>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
