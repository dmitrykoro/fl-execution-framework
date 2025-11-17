# 🚀 Frontend

A React UI for the `fl-execution-framework` - built with React 19, Vite, React Bootstrap, and Recharts.

## ✨ Features

### 📊 **Dashboard & Visualization**

- **Interactive Dashboard**: Browse all simulation runs with real-time status updates
- **Simulation Details**: Inspect configuration, results, and metrics for any run
- **Comparison View**: Side-by-side comparison of multiple simulations with performance metrics
- **Interactive Plots** (Recharts):
  - Per-client metric visualization with toggle controls
  - Round-level aggregate metrics
  - Malicious client highlighting
  - Zoom & brush controls for detailed analysis
  - Theme-aware chart styling (dark/light mode)

### 🎯 **Simulation Configuration**

- **New Simulation Form**: Comprehensive UI for launching federated learning simulations
- **Real-time Validation**:
  - HuggingFace dataset validation with metadata preview
  - Strategy-specific validation (Krum, Multi-Krum, Bulyan constraints)
  - Model-dataset compatibility checks (CNN vs Transformer)
  - Configuration error detection before launch
- **Dataset Loading**: Direct integration with HuggingFace Hub
  - IID, Dirichlet, and Pathological partitioning strategies
  - Real-time dataset validation with split/feature info
  - Example count formatting and key features display
- **Dynamic Poisoning Attacks**:
  - Round-based attack scheduling
  - Label flipping, Gaussian noise, and brightness attacks
  - Attack ratio configuration

### 🎨 **UI/UX Features**

- **Material Design 3**: Modern dark/light theme with custom color tokens
- **Theme Toggle**: Persistent theme preference with system detection
- **Error Boundaries**: Graceful error handling throughout the app
- **Responsive Design**: Bootstrap-based responsive layout
- **Validation Summary**: Clear error/warning/info messages with emoji indicators
- **Auto-save Drafts**: localStorage persistence for simulation configurations

## 🔌 API Integration

The frontend communicates with a **FastAPI backend** providing:

### **REST Endpoints**

- `GET /api/simulations` - List all simulation runs with metadata
- `GET /api/simulations/{id}` - Get simulation details and result files
- `GET /api/simulations/{id}/status` - Poll simulation status (running/completed/failed)
- `GET /api/simulations/{id}/plot-data` - Fetch interactive plot data (JSON)
- `GET /api/simulations/{id}/results/{filename}` - Download result files (PNG, PDF, CSV, JSON)
- `POST /api/simulations` - Create and launch new simulation
- `GET /api/datasets/validate` - Validate HuggingFace dataset (real-time validation)

### **Features**

- **CORS**: Configured for ports 5173-5178 (Vite dev server auto-increment)
- **Security**: Path traversal protection with `secure_join`
- **Error Handling**: Categorized error messages (network, 404, auth, forbidden)
- **CSV Parsing**: Latin-1 encoding support for legacy datasets
- **Process Management**: Background simulation execution with status tracking

## 🛠️ Tech Stack

### **Frontend**

- **React 19.1** - Latest React with improved hooks and performance
- **Vite 7.1** - Fast build tool and dev server
- **React Bootstrap 2.10** - UI component library (Material Design 3 themed)
- **Recharts 3.2** - Interactive chart library for metrics visualization
- **React Router 7.9** - Client-side routing
- **Axios 1.12** - HTTP client for API requests
- **Lucide React 0.544** - Icon library

### **Development Tools**

- **ESLint** - Code linting with React-specific rules
- **Prettier** - Code formatting
- **Vite Plugin React** - Fast Refresh and JSX support

## 📚 Documentation

### **Quick Links**

- **[Dataset Loading](./datasets.md)** - HuggingFace datasets with IID/Dirichlet/Pathological partitioning
- **[Attack Scheduling](./attacks.md)** - Dynamic round-based poisoning attacks

### **Documentation Map** 🗺️

```text
frontend/docs/
├── README.md       ← You are here! Frontend overview
├── datasets.md     ← Dataset loading guide
│   ├── IID partitioning
│   ├── Dirichlet (heterogeneous)
│   ├── Pathological (extreme non-IID)
│   └── Real-time validation API
└── attacks.md      ← Attack scheduling guide
    ├── Label flipping
    ├── Gaussian noise
    ├── Brightness attacks
    └── Multi-phase scheduling
```

**What to read:**

- **New to the project?** Start with this README, then explore dataset/attack docs
- **Configuring datasets?** [datasets.md](./datasets.md)
- **Setting up attacks?** [attacks.md](./attacks.md)
- **API integration?** See [API Integration](#-api-integration) section above

## 🛠️ Running the Application

### Quick Start (Recommended)

From the project root, run the startup script that handles both servers:

```bash
./run_frontend.sh
```

This automatically:

- ✅ Installs dependencies (Python + npm)
- ✅ Starts API server (port 8000)
- ✅ Starts frontend dev server (port 5173)
- ✅ Opens browser to `http://localhost:5173`
- ✅ Saves logs to `tests/logs/`

Press `Ctrl+C` to stop both servers.

### GPU Acceleration (Optional)

For faster training (3-10x speedup with CUDA-enabled GPUs):

**Requirements:**

- NVIDIA GPU (e.g., RTX 3060 Ti or better)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- Updated NVIDIA drivers

**Setup:**

After running `./reinstall_requirements.sh`, install PyTorch with CUDA support:

```bash
pip uninstall torch torchvision -y
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

**Verify:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:

```bash
PyTorch: 2.2.2+cu118
CUDA: True
```

The framework automatically detects and uses GPU when available.

### Manual Setup (Alternative)

Requires two separate terminals.

#### 1. Start API

From the project root, install API dependencies and start the server:

```bash
# Install API-specific dependencies
pip install -r src/api/requirements.txt

# Start the backend server
python -m uvicorn src.api.main:app --reload --port 8000
```

#### 2. Start Frontend

From the `frontend/` directory, start the UI:

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` in your browser.

## 📁 Project Structure

```text
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── Dashboard.jsx           # Main simulation list view
│   │   ├── SimulationDetails.jsx   # Individual simulation results
│   │   ├── NewSimulation.jsx       # Simulation configuration form
│   │   ├── ComparisonView.jsx      # Multi-simulation comparison
│   │   ├── InteractivePlots.jsx    # Recharts visualizations
│   │   ├── RoundMetricsPlot.jsx    # Round-level aggregate metrics
│   │   ├── ValidationSummary.jsx   # Error/warning display
│   │   ├── ThemeToggle.jsx         # Dark/light mode switcher
│   │   └── ErrorBoundary.jsx       # Error handling wrapper
│   ├── hooks/                # Custom React hooks
│   │   ├── useApi.js               # API request abstraction
│   │   ├── useConfigValidation.js  # Real-time config validation
│   │   └── useDatasetValidation.js # HuggingFace dataset validation
│   ├── contexts/             # React contexts
│   │   └── ThemeContext.jsx        # Theme state management
│   ├── utils/                # Utility functions
│   │   └── configValidation.js     # Strategy validation logic
│   ├── api.js                # API client (Axios)
│   ├── App.jsx               # Root component with routing
│   ├── App.css               # Material Design 3 theming
│   ├── index.css             # Global styles & MD3 tokens
│   └── main.jsx              # React entry point
├── docs/                     # Documentation
│   ├── README.md             # This file
│   ├── HUGGINGFACE_DATASETS.md
│   └── DYNAMIC_POISONING.md
├── public/                   # Static assets
├── package.json              # Dependencies & scripts
└── vite.config.js            # Vite configuration
```

## 🎯 Key Components

### **Dashboard.jsx**

- Lists all simulations with sortable table
- Real-time status badges (running/completed/failed)
- Multi-select for comparison view
- Search and filter capabilities

### **NewSimulation.jsx**

- Comprehensive form with 30+ configuration options
- Real-time validation with debounced API calls
- localStorage draft persistence
- Strategy-specific parameter visibility
- HuggingFace dataset autocomplete with validation

### **InteractivePlots.jsx**

- Client-level metric visualization
- Toggleable client lines with malicious highlighting
- Zoom/brush controls for detailed analysis
- Theme-aware Recharts styling
- Metric selector dropdown

### **ComparisonView.jsx**

- Side-by-side simulation comparison
- Configuration diff highlighting
- Performance metrics comparison table
- Simultaneous plot viewing

### **ValidationSummary.jsx**

- Categorized validation messages (errors, warnings, info)
- Emoji indicators for quick scanning
- Detailed field-level error messages
- Launch button state control

## 🔧 Development

### **Available Scripts**

```bash
npm run dev        # Start Vite dev server (port 5173)
npm run build      # Production build
npm run preview    # Preview production build
npm run lint       # Run ESLint
npm run format     # Format code with Prettier
npm run css:audit  # Analyze CSS for unused classes
```

### **Environment**

- **API URL**: Hardcoded to `http://localhost:8000` (development)
- **Hot Reload**: Enabled via Vite Fast Refresh
- **Port**: 5173 (auto-increments if occupied: 5174-5178)
