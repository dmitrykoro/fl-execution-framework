# 🚀 Frontend

A React UI for the `fl-execution-framework`.

## ✨ Features

- **Dashboard**: View past simulation runs.
- **Details**: Inspect configuration and results.
- **Execution**: Launch new simulations from the UI.

## 🛠️ Running the Application

Requires two separate terminals.

### 1. Start API

From the project root, install API dependencies and start the server:

```bash
# Install API-specific dependencies
pip install -r src/api/requirements.txt

# Start the backend server
uvicorn src.api.main:app --reload
```

### 2. Start Frontend

From the `frontend/` directory, start the UI:

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` in your browser.
