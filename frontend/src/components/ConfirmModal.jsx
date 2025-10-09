import { Modal, Button } from 'react-bootstrap';
import OutlineButton from './OutlineButton';

/**
 * Reusable confirmation modal component with Material Design 3 styling
 *
 * @param {boolean} show - Whether to show the modal
 * @param {string} title - Modal title text
 * @param {string} message - Confirmation message (can include \n for multiline)
 * @param {function} onConfirm - Callback when user confirms
 * @param {function} onCancel - Callback when user cancels
 * @param {string} variant - Bootstrap variant for confirm button (danger, warning, primary, info)
 * @param {string} confirmText - Text for confirm button (default: "Confirm")
 * @param {string} cancelText - Text for cancel button (default: "Cancel")
 */
function ConfirmModal({
  show,
  title,
  message,
  onConfirm,
  onCancel,
  variant = 'danger',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
}) {
  return (
    <Modal show={show} onHide={onCancel} centered className="confirm-modal">
      <Modal.Header closeButton>
        <Modal.Title>{title}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p style={{ whiteSpace: 'pre-line', marginBottom: 0 }}>{message}</p>
      </Modal.Body>
      <Modal.Footer>
        <OutlineButton onClick={onCancel}>{cancelText}</OutlineButton>
        <Button variant={variant} onClick={onConfirm}>
          {confirmText}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default ConfirmModal;
