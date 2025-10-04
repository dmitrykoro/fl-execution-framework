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

# Create log files
API_LOG=$(mktemp)
FRONTEND_LOG=$(mktemp)

# Start API in background with logging
uvicorn src.api.main:app --reload --port 8000 > "$API_LOG" 2>&1 &
API_PID=$!

# Start frontend in background with logging
cd frontend
npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for servers to start, then open browser
sleep 3
if command -v xdg-open > /dev/null; then
  xdg-open http://localhost:5173 2>/dev/null
elif command -v open > /dev/null; then
  open http://localhost:5173 2>/dev/null
elif command -v start > /dev/null; then
  start http://localhost:5173 2>/dev/null
fi

# Trap Ctrl+C to kill both processes and cleanup
trap "echo ''; echo '🛑 Stopping servers...'; kill $API_PID $FRONTEND_PID 2>/dev/null; rm -f $API_LOG $FRONTEND_LOG; exit" INT TERM

echo ""
echo "📋 Tailing logs (Ctrl+C to stop)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Tail both logs with prefixes
tail -f "$API_LOG" "$FRONTEND_LOG" 2>/dev/null | while IFS= read -r line; do
  if [[ "$line" == *"==> $API_LOG <=="* ]]; then
    echo -e "\n🔵 [API]"
  elif [[ "$line" == *"==> $FRONTEND_LOG <=="* ]]; then
    echo -e "\n🟢 [FRONTEND]"
  elif [[ -n "$line" ]]; then
    echo "$line"
  fi
done