import { Accordion, Table, Spinner } from 'react-bootstrap';
import OutlineButton from '@components/common/Button/OutlineButton';
import { InfoTooltip } from '@components/common/Tooltip/InfoTooltip';
import { copyCSVToClipboard, filterEmptyColumns, formatColumnName } from '@utils/csvHelpers';
import { getMetricTooltip } from '@constants/metricTooltips';

export function RawDataAccordion({ csvFiles, csvData, simulationId }) {
  if (!csvFiles || csvFiles.length === 0) {
    return null;
  }

  return (
    <Accordion>
      <Accordion.Item eventKey="raw">
        <Accordion.Header>Advanced: Raw CSV Data (for export/analysis)</Accordion.Header>
        <Accordion.Body>
          {csvFiles.map(file => {
            const data = csvData[file];
            if (!data || data.length === 0) {
              return (
                <div key={file} className="mb-4">
                  <h6 className="text-muted">{file}</h6>
                  <Spinner animation="border" size="sm" />
                </div>
              );
            }
            const columns = filterEmptyColumns(data);
            const downloadUrl = `/api/simulations/${simulationId}/results/${file}?download=true`;

            return (
              <div key={file} className="mb-4">
                <div className="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center gap-2 mb-2">
                  <h6 className="text-muted font-monospace small mb-0">{file}</h6>
                  <div className="d-flex flex-column flex-sm-row gap-2 w-100 w-md-auto">
                    <OutlineButton
                      as="a"
                      href={downloadUrl}
                      download={file.split('/').pop()}
                      title="Download CSV file to your computer"
                      className="d-flex align-items-center justify-content-center"
                      style={{
                        padding: '0.25rem 0.5rem',
                        minWidth: 'auto',
                        fontSize: '0.875rem',
                        border: 'none',
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                        download
                      </span>
                    </OutlineButton>
                    <OutlineButton
                      onClick={() => copyCSVToClipboard(data, file)}
                      title="Copy data to clipboard for pasting into Excel/Google Sheets"
                      className="d-flex align-items-center justify-content-center"
                      style={{
                        padding: '0.25rem 0.5rem',
                        minWidth: 'auto',
                        fontSize: '0.875rem',
                        border: 'none',
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                        content_copy
                      </span>
                    </OutlineButton>
                  </div>
                </div>
                <div style={{ overflowX: 'auto', fontSize: '0.75rem' }}>
                  <Table striped bordered hover size="sm">
                    <thead>
                      <tr>
                        {columns.map(col => {
                          const formattedName = formatColumnName(col);
                          const tooltip = getMetricTooltip(col);

                          return (
                            <th key={col}>
                              {tooltip ? (
                                <InfoTooltip text={tooltip} placement="top">
                                  {formattedName}
                                </InfoTooltip>
                              ) : (
                                formattedName
                              )}
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {data.map((row, idx) => (
                        <tr key={idx}>
                          {columns.map(col => (
                            <td key={col} className="font-monospace">
                              {row[col]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </div>
              </div>
            );
          })}
        </Accordion.Body>
      </Accordion.Item>
    </Accordion>
  );
}
