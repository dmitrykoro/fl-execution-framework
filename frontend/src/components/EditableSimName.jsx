import { useState } from 'react';
import { Form, Button, Spinner } from 'react-bootstrap';
import { renameSimulation } from '../api';

/**
 * Inline editable simulation name component
 * Click to edit, blur/Enter to save, Escape to cancel
 */
function EditableSimName({ simulationId, displayName, onRename }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(displayName || '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const handleEdit = () => {
    setEditValue(displayName || simulationId);
    setIsEditing(true);
    setError(null);
  };

  const handleCancel = () => {
    setEditValue(displayName || '');
    setIsEditing(false);
    setError(null);
  };

  const handleSave = async () => {
    const trimmed = editValue.trim();

    if (!trimmed) {
      setError('Name cannot be empty');
      return;
    }

    if (trimmed.length > 100) {
      setError('Name must be 100 characters or less');
      return;
    }

    if (!/^[a-zA-Z0-9\s\-_]+$/.test(trimmed)) {
      setError('Only letters, numbers, spaces, hyphens, and underscores allowed');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await renameSimulation(simulationId, trimmed);
      onRename(trimmed);
      setIsEditing(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to rename');
    } finally {
      setSaving(false);
    }
  };

  const handleKeyPress = e => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  const handleBlur = () => {
    if (!saving) {
      handleSave();
    }
  };

  if (isEditing) {
    return (
      <div>
        <Form.Control
          type="text"
          value={editValue}
          onChange={e => setEditValue(e.target.value)}
          onKeyDown={handleKeyPress}
          onBlur={handleBlur}
          placeholder="Enter simulation name"
          maxLength={100}
          autoFocus
          disabled={saving}
          isInvalid={!!error}
          size="sm"
        />
        {error && (
          <Form.Control.Feedback type="invalid" className="d-block">
            {error}
          </Form.Control.Feedback>
        )}
      </div>
    );
  }

  return (
    <span
      onClick={handleEdit}
      className="mb-0"
      style={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        display: 'block',
        minWidth: 0,
      }}
      title="Click to rename"
    >
      {displayName || simulationId}
    </span>
  );
}

export default EditableSimName;
