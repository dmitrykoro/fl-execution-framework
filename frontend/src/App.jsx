import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { Dashboard } from './pages/Dashboard/Dashboard';
import { SimulationDetails } from './pages/SimulationDetails/SimulationDetails';
import { NewSimulation } from './pages/NewSimulation/NewSimulation';
import ComparisonView from './components/ComparisonView';
import ErrorBoundary from './components/ErrorBoundary';
import ThemeToggle from './components/ThemeToggle';
import { ToastProvider } from './components/ToastProvider';
import './App.css';

function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
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
              <Route path="/simulations/:simulationId" element={<SimulationDetails />} />
              <Route path="/compare" element={<ComparisonView />} />
            </Routes>
          </Container>
        </Router>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
