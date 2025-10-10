import { useState, useCallback } from 'react';
import { Alert } from 'react-bootstrap';
import { MaterialIcon } from './common/Icon/MaterialIcon';
import { ToastContext } from '../contexts/ToastContext';

/**
 * Toast provider component that manages global toast notifications
 */
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const removeToast = useCallback(id => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  }, []);

  const showToast = useCallback(
    (message, variant = 'info', duration = 4000) => {
      const id = Date.now() + Math.random();
      const toast = { id, message, variant };

      setToasts(prev => [...prev, toast]);

      if (duration > 0) {
        setTimeout(() => removeToast(id), duration);
      }

      return id;
    },
    [removeToast]
  );

  const showSuccess = useCallback(
    (message, duration) => showToast(message, 'success', duration),
    [showToast]
  );
  const showError = useCallback(
    (message, duration) => showToast(message, 'danger', duration),
    [showToast]
  );
  const showWarning = useCallback(
    (message, duration) => showToast(message, 'warning', duration),
    [showToast]
  );
  const showInfo = useCallback(
    (message, duration) => showToast(message, 'info', duration),
    [showToast]
  );

  const getIcon = variant => {
    switch (variant) {
      case 'success':
        return <MaterialIcon name="check_circle" size={20} fill={1} />;
      case 'danger':
        return <MaterialIcon name="cancel" size={20} fill={1} />;
      case 'warning':
        return <MaterialIcon name="warning" size={20} fill={1} />;
      case 'info':
      default:
        return <MaterialIcon name="info" size={20} fill={1} />;
    }
  };

  return (
    <ToastContext.Provider value={{ showToast, showSuccess, showError, showWarning, showInfo }}>
      {children}
      <div className="toast-container">
        {toasts.map(toast => (
          <Alert
            key={toast.id}
            variant={toast.variant}
            className="toast-notification"
            onClose={() => removeToast(toast.id)}
            dismissible
          >
            <div className="d-flex align-items-center gap-2">
              <span className="toast-icon">{getIcon(toast.variant)}</span>
              <span className="toast-message">{toast.message}</span>
            </div>
          </Alert>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
