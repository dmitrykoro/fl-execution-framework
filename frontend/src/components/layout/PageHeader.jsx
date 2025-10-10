export function PageHeader({ title, action, subtitle, children }) {
  return (
    <div className="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-4 gap-3">
      <div>
        <h1 className="mb-0">{title}</h1>
        {subtitle && <p className="text-muted mb-0 mt-1">{subtitle}</p>}
      </div>
      {(action || children) && <div>{action || children}</div>}
    </div>
  );
}
