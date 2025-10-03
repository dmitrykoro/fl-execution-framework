#!/bin/bash
# Dev server startup script

set -e

echo "🚀 Starting FL Framework Development Servers..."
echo ""

# Install Python API dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting servers..."
echo "  - API: http://127.0.0.1:8000"
echo "  - Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start API in background
uvicorn src.api.main:app --reload &
API_PID=$!

# Start frontend (foreground)
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for servers to start, then open browser
sleep 3
if command -v xdg-open > /dev/null; then
  xdg-open http://localhost:5173
elif command -v open > /dev/null; then
  open http://localhost:5173
elif command -v start > /dev/null; then
  start http://localhost:5173
fi

# Trap Ctrl+C to kill both processes
trap "echo ''; echo '🛑 Stopping servers...'; kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Wait for both processes
wait