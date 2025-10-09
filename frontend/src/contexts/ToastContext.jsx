import { createContext, useContext } from 'react';

/**
 * Toast context for managing global notifications
 */
export const ToastContext = createContext(null);

/**
 * Custom hook to access toast notifications
 * @returns {Object} Toast functions: showToast, showSuccess, showError, showWarning, showInfo
 */
export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return context;
};
