import { Navbar as BSNavbar, Nav, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import ThemeToggle from '../ThemeToggle';

export function Navbar() {
  return (
    <BSNavbar
      expand="md"
      className="mb-3"
      style={{
        backgroundColor: 'var(--md-sys-color-surface-variant)',
        color: 'var(--md-sys-color-primary)',
      }}
    >
      <Container>
        <BSNavbar.Brand as={Link} to="/" style={{ color: 'var(--md-sys-color-primary)' }}>
          FL Framework
        </BSNavbar.Brand>
        <BSNavbar.Toggle aria-controls="navbar-nav" />
        <BSNavbar.Collapse id="navbar-nav">
          <Nav className="me-auto">
            <Nav.Link as={Link} to="/" style={{ color: 'var(--md-sys-color-primary)' }}>
              Dashboard
            </Nav.Link>
            <Nav.Link
              as={Link}
              to="/simulations/new"
              style={{ color: 'var(--md-sys-color-primary)' }}
            >
              New Simulation
            </Nav.Link>
          </Nav>
          <Nav>
            <ThemeToggle />
          </Nav>
        </BSNavbar.Collapse>
      </Container>
    </BSNavbar>
  );
}
