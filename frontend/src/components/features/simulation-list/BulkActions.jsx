import { Button } from 'react-bootstrap';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';

export function BulkActions({
  selectedCount,
  totalCount,
  onDeleteSelected,
  onCompare,
  onClearAll,
  deleting,
}) {
  return (
    <div className="d-flex flex-wrap gap-2 w-100 w-md-auto">
      {selectedCount > 0 && (
        <>
          <Button
            variant="danger"
            size="sm"
            onClick={onDeleteSelected}
            disabled={deleting}
            className="flex-grow-1 flex-md-grow-0"
          >
            🗑️ Delete ({selectedCount})
          </Button>
          <Button
            variant="info"
            size="sm"
            onClick={onCompare}
            className="flex-grow-1 flex-md-grow-0"
          >
            📊 Compare ({selectedCount})
          </Button>
        </>
      )}
      {totalCount > 0 && (
        <Button
          variant="outline-danger"
          size="sm"
          onClick={onClearAll}
          disabled={deleting}
          className="flex-grow-1 flex-md-grow-0"
        >
          <div className="d-flex align-items-center gap-1">
            <MaterialIcon name="delete" size={20} />
            <span>Clear All</span>
          </div>
        </Button>
      )}
    </div>
  );
}
