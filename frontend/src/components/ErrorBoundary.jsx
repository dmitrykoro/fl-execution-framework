import React from 'react';
import { Alert, Button, Container } from 'react-bootstrap';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      return (
        <Container className="mt-5">
          <Alert variant="danger">
            <Alert.Heading>Something went wrong</Alert.Heading>
            <p>
              The application encountered an unexpected error. Please try returning to the
              dashboard.
            </p>
            <hr />
            <details className="mb-3">
              <summary style={{ cursor: 'pointer' }}>Error details</summary>
              <pre className="mt-2 p-2 bg-light border rounded">
                {this.state.error && this.state.error.toString()}
              </pre>
            </details>
            <Button variant="primary" onClick={this.handleReset}>
              Return to Dashboard
            </Button>
          </Alert>
        </Container>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
