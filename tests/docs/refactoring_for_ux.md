# Technical Summary of Web UI and API Development

## 📜 Summary

A complete web-based user interface was developed to provide an intuitive way to configure, launch, monitor, and analyze federated learning simulations. This required creating a new FastAPI backend server, a React frontend application, and supporting infrastructure. The core simulation logic remains unchanged - this enhancement provides a modern interface layer over the existing command-line framework.

## 🎯 The Purpose of the Web UI

The web UI transforms the federated learning framework from a CLI-only tool into an accessible platform:

- **Easy Configuration**: Visual forms replace complex JSON file editing
- **Real-time Monitoring**: Live status updates during simulation execution
- **Result Visualization**: Interactive charts and downloadable result files
- **Historical Tracking**: Browse and compare past simulation runs
- **Lower Barrier to Entry**: Researchers can focus on experiments, not command syntax

---

## 🏗️ Architecture Overview

The web UI follows a modern three-tier architecture:

1. **Frontend (React)**: User interface for configuration and visualization
2. **Backend (FastAPI)**: REST API server for simulation management
3. **Simulation Layer**: Existing Python federated learning framework

```text
┌─────────────────┐
│  React Frontend │ (Port 5173)
│   (Vite + Axios)│
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Server │ (Port 8000)
│   (CORS enabled)│
└────────┬────────┘
         │ subprocess
         ▼
┌─────────────────┐
│  FL Simulation  │
│  (run_simulation│
│     .sh)        │
└─────────────────┘
```

---

## 🛠️ Backend Development (FastAPI)

### File: `src/api/main.py` (New File - 350+ lines)

**Core Components:**

1. **CORS Configuration**
   - Allows frontend (localhost:5173) to communicate with API
   - Development-friendly settings (all methods/headers allowed)
   - **Risk**: Very low - standard dev configuration

2. **Pydantic Models**
   - `SimulationConfig`: 40+ configurable parameters
   - `SimulationMetadata`: Summary info for list views
   - `SimulationDetails`: Full config + result files
   - **Purpose**: Type-safe request/response validation

3. **Path Security**
   - `secure_join()`: Prevents path traversal attacks
   - `get_simulation_path()`: Validates simulation IDs
   - **Purpose**: Protect filesystem from malicious inputs

4. **Process Management**
   - `running_processes`: In-memory dict of active simulations
   - Tracks subprocess handles for status monitoring
   - **Purpose**: Enable real-time progress tracking

### API Endpoints

#### `GET /api/simulations`

- **Purpose**: List all historical simulation runs
- **Returns**: Array of simulation metadata (ID, strategy, rounds, clients)
- **Implementation**: Scans `out/` directory for config.json files

#### `POST /api/simulations`

- **Purpose**: Launch a new simulation
- **Input**: SimulationConfig with all parameters
- **Process**:
  1. Generate unique simulation ID (timestamp-based)
  2. Create output directory
  3. Write config.json
  4. Launch run_simulation.sh as subprocess
  5. Store process handle for monitoring
- **Returns**: Simulation ID and status

#### `GET /api/simulations/{id}`

- **Purpose**: Get detailed info for a specific simulation
- **Returns**: Full config + list of result files (CSVs, PNGs, etc.)

#### `GET /api/simulations/{id}/status`

- **Purpose**: Real-time status monitoring
- **Returns**: `{status: "pending|running|completed|failed", progress: 0.0-1.0}`
- **Logic**:
  - Checks if process is in `running_processes` dict
  - Polls process with `poll()` to see if still active
  - Falls back to checking for result files if process not tracked
  - Handles server restarts gracefully

#### `GET /api/results/{filename}`

- **Purpose**: Download result files (CSVs, plots)
- **Security**: Validates filename, prevents directory traversal
- **Returns**: File download or JSON metadata

### Development Script: `start-dev.sh` (New File)

**Purpose**: Convenient parallel startup of frontend and backend

```bash
#!/bin/bash
# Start FastAPI server and React frontend concurrently
uvicorn src.api.main:app --reload &
cd frontend && npm run dev
```

**Usage**: `./start-dev.sh` from project root

---

## 🎨 Frontend Development (React + Vite)

### Project Structure

```text
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx          # Home page with simulation list
│   │   ├── NewSimulation.jsx      # Configuration form
│   │   ├── SimulationDetails.jsx  # Results viewer
│   │   └── ErrorBoundary.jsx      # Error handling wrapper
│   ├── hooks/
│   │   └── useApi.js              # Custom React hooks for API calls
│   ├── api.js                     # Axios HTTP client
│   ├── App.jsx                    # Main routing component
│   └── main.jsx                   # React entry point
├── index.html
├── package.json                   # Dependencies
└── vite.config.js                 # Build configuration
```

### Key Dependencies

```json
{
  "react": "^18.3.1",
  "react-router-dom": "^7.1.3",  // Client-side routing
  "axios": "^1.7.9",              // HTTP requests
  "recharts": "^2.15.0",          // Chart visualizations
  "vite": "^6.0.5"                // Build tool
}
```

### Component Details

#### `Dashboard.jsx`

- **Purpose**: Landing page showing all simulations
- **Features**:
  - Fetches simulation list from API on mount
  - Displays as sortable table
  - Filters by strategy, status, date
  - "New Simulation" button navigates to form
- **State Management**: React hooks (useState, useEffect)

#### `NewSimulation.jsx`

- **Purpose**: Interactive form for simulation configuration
- **Features**:
  - 40+ form fields organized into sections:
    - Basic Settings (strategy, dataset, rounds)
    - Client Configuration (count, malicious clients, attack type)
    - Advanced Options (hyperparameters, device settings)
  - Conditional fields (e.g., PID params only show for PID strategy)
  - Form validation (required fields, numeric ranges)
  - Submit triggers POST to `/api/simulations`
- **UX**: Clear labels, tooltips, default values

#### `SimulationDetails.jsx`

- **Purpose**: View results for a completed simulation
- **Features**:
  - Displays full configuration as formatted JSON
  - Lists all output files with download buttons
  - Shows plots inline (accuracy, loss, client metrics)
  - Interactive charts with hover tooltips (Recharts)
  - Status polling for in-progress simulations
- **Auto-refresh**: Polls `/api/simulations/{id}/status` every 3 seconds

#### `ErrorBoundary.jsx`

- **Purpose**: Graceful error handling
- **Implementation**: React error boundary component
- **Displays**: User-friendly message instead of white screen

### API Client: `api.js`

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' }
});

export const getSimulations = () => api.get('/api/simulations');
export const createSimulation = (config) => api.post('/api/simulations', config);
export const getSimulationDetails = (id) => api.get(`/api/simulations/${id}`);
export const getSimulationStatus = (id) => api.get(`/api/simulations/${id}/status`);
```

### Custom Hook: `useApi.js`

```javascript
export function useSimulations() {
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getSimulations()
      .then(res => setSimulations(res.data))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return { simulations, loading, error };
}
```

**Purpose**: Encapsulate API calls with loading/error states

---

## 🔧 Infrastructure Changes

### Root-Level Files

#### `package.json` (New File)

- **Purpose**: Node.js workspace configuration
- **Key Scripts**:
  - `dev`: Start both frontend and backend
  - `build`: Production build of frontend
  - `test`: Run frontend tests

#### `package-lock.json` (Auto-generated)

- **Purpose**: Lock dependency versions for reproducibility

#### `.gitignore` Updates

- Added `frontend/node_modules/`
- Added `frontend/dist/`
- Added `frontend/.env`

### Backend Integration

#### `src/federated_simulation.py` (Modifications)

**FedAvg Strategy Support:**

- **Lines 358-367**: Added FedAvg initialization case
- **Purpose**: Enable vanilla Flower FedAvg as baseline strategy
- **Context**: Frontend defaults to FedAvg for simplicity
- **Risk**: Very low - 12 lines following existing patterns

#### `src/config_loaders/validate_strategy_config.py` (Modifications)

**FedAvg Validation:**

- **Line 10**: Added "FedAvg" to strategy enum
- **Purpose**: Allow FedAvg in config validation
- **Risk**: Very low - single enum entry

#### `src/client_models/flower_client.py` (Bug Fixes)

**Sample Count Corrections:**

- **Line 201**: Fixed `fit()` to return dataset size, not batch count
- **Line 206-207**: Fixed `evaluate()` to return dataset size
- **Problem**:
  - `len(self.trainloader)` returns batch count
  - `len(self.valloader)` returns batch count
  - Flower framework needs sample counts for weighted averaging
  - Incorrect counts caused ZeroDivisionError
- **Fix**: Use `len(loader.dataset)` with fallback to batch count
- **Risk**: Low - critical bug fix, 2 lines changed

---

## 📊 Python 3.9-3.11 Compatibility & API Details

### Union Type Syntax Fix

**File**: `src/api/main.py`

**Changes**:

- `int | str` → `Union[int, str]` (Line 94)
- `FileResponse | JSONResponse` → `Union[FileResponse, JSONResponse]`
- Added `response_model=None` to `/results/{filename}` endpoint

**Reason**: PEP 604 union syntax (`X | Y`) requires Python 3.10+

**Impact**: Ensures compatibility with Python 3.9-3.11

**Risk**: Very low - syntax-only change

### CORS Configuration Details

**File**: `src/api/main.py` (Lines 22-28)

**Implementation**:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Purpose**: Allow React frontend to communicate with FastAPI backend during development

**Security**: Development-only configuration; production should restrict origins

### Process Tracking Implementation

**File**: `src/api/main.py` (Line 37)

**Implementation**:

```python
# In-memory storage for running processes
running_processes: Dict[str, subprocess.Popen] = {}
```

**Usage**:

- Store process handle when launching simulation: `running_processes[sim_id] = subprocess.Popen(...)`
- Check status via `process.poll()` - returns None if running, exit code if done
- Log PID for debugging: `logger.info(f"Started simulation {sim_id} with PID {process.pid}")`

**Limitation**: Process handles lost on server restart (gracefully handled by status endpoint)

### Status Endpoint Logic

**File**: `src/api/main.py`

**Endpoint**: `GET /api/simulations/{id}/status`

**Response Model**:

```json
{
  "status": "pending|running|completed|failed",
  "progress": 0.0  // 0.0 to 1.0
}
```

**Logic Flow**:

1. Check if `sim_id` exists in `running_processes` dict
2. If yes, call `process.poll()`:
   - Returns `None` → status = "running"
   - Returns non-zero → status = "failed"
   - Returns 0 → status = "completed"
3. If not tracked, check filesystem:
   - Results exist → status = "completed"
   - No results → status = "pending" (or failed if old)
4. Gracefully handles server restarts by falling back to file detection

**Risk**: Very low - read-only, no state modification

---

## 🎨 Styling and UX Polish

### CSS Framework: Tailwind CSS (Potential)

- **Status**: Not yet implemented
- **Alternative**: Inline styles or CSS modules

### Responsive Design

- **Mobile**: Not yet optimized
- **Desktop**: Functional layout with flexbox/grid

### Loading States

- Spinners during API calls
- Skeleton screens for data fetching

### Error Handling

- User-friendly error messages
- Toast notifications for success/failure

---

## 🚀 Deployment Considerations

### Current State: Development Mode

**Frontend**:

- Vite dev server (port 5173)
- Hot module replacement enabled

**Backend**:

- Uvicorn with `--reload` flag
- Auto-restart on file changes

### Production Readiness (Future Work)

**Frontend**:

- Build with `npm run build` → static files in `frontend/dist/`
- Serve via Nginx or CDN

**Backend**:

- Run uvicorn without `--reload`
- Use Gunicorn for multi-worker support
- Add authentication (JWT tokens)
- Rate limiting for API endpoints

**Security Enhancements**:

- CORS: Restrict to specific origins
- API Keys: Require authentication
- Input Validation: Stricter parameter checks
- File Upload: Validate dataset uploads

---

## 📈 Impact Summary

**✅ Key Outcomes:**

- 🎨 **Modern UI**: Intuitive web interface replaces CLI-only access
- 📊 **Real-time Monitoring**: Live status updates during simulations
- 📁 **Result Management**: Easy access to plots, CSVs, and configs
- 🔄 **Historical Tracking**: Browse and compare past experiments
- 🛡️ **Type Safety**: Pydantic models ensure API correctness
- 🌐 **API-First Design**: Backend can support multiple frontends

**📝 Developer Guidelines:**

- 🔧 **Start Dev Environment**: Run `./start-dev.sh` or manually start both servers
- 🌐 **Frontend URL**: <http://localhost:5173>
- 🔌 **Backend URL**: <http://localhost:8000>
- 📚 **API Docs**: <http://localhost:8000/docs> (auto-generated by FastAPI)

---

## 📊 File Statistics

**New Files Created:**

- Backend: 1 file (main.py, ~350 lines)
- Frontend: 8+ files (~1500 lines total)
- Infrastructure: 2 files (package.json, start-dev.sh)

**Modified Files:**

- `src/federated_simulation.py`: +12 lines (FedAvg support)
- `src/config_loaders/validate_strategy_config.py`: +1 line (FedAvg enum)
- `src/client_models/flower_client.py`: ~5 lines (sample count fixes)
- **Code Formatting**: 48 Python files reformatted with ruff (1693 insertions, 1080 deletions)

**Total New Code**: ~2000 lines (excluding node_modules)

---

## 🔮 Future Enhancements

**Short Term:**

- Add authentication (user login)
- Implement dataset upload via UI
- Add simulation deletion/cancellation
- Show live logs during execution

**Medium Term:**

- Multi-user support with role-based access
- Simulation comparison dashboard
- Export results to PDF reports
- Scheduled/recurring simulations

**Long Term:**

- Distributed execution across multiple machines
- Cloud deployment (AWS/Azure/GCP)
- Hyperparameter tuning suggestions
- Integration with MLflow/Weights & Biases

---

## 🏁 Conclusion

The web UI represents a major usability enhancement to the federated learning framework. By providing a modern, intuitive interface, we've made the system accessible to a broader range of researchers and practitioners. The API-first design ensures flexibility for future integrations, while the clean separation of concerns maintains the integrity of the core simulation logic.

**Core Principles Maintained:**

- ✅ No changes to core FL algorithms
- ✅ Backward compatibility with CLI usage
- ✅ Output format unchanged (CSVs, plots)
- ✅ Configuration schema preserved

**Quality Improvements:**

- 📈 Enhanced accessibility for non-technical users
- 🔍 Better experiment tracking and reproducibility
- 🎨 Improved visualization of results
- 🤝 Foundation for collaborative research platform

---

Happy experimenting! 🎓🚀
