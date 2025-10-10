import { Spinner } from './Spinner';

export function LoadingPage({ message = 'Loading...' }) {
  return (
    <div
      className="d-flex flex-column align-items-center justify-content-center"
      style={{ minHeight: '300px' }}
    >
      <Spinner size="lg" />
      {message && <p className="mt-3 text-muted">{message}</p>}
    </div>
  );
}
