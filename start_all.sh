#!/bin/bash

# Script to start the Flask backend server & Frontend in the background

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create it first:"
    echo "  python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Start Flask backend server in background with nohup
echo "Starting Flask Backend engine..."
nohup python3 vapi_controller.py > log/backend.log 2>&1 &

# Save the backend process ID
BACKEND_PID=$!
echo $BACKEND_PID > log/backend.pid

echo "Backend server started in background"
echo "  PID: $BACKEND_PID"
echo "  Logs: log/backend.log"
echo ""

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo "Error: Backend server failed to start. Check log/backend.log"
    exit 1
fi

# Start frontend
echo "======================================"
echo "Starting frontend..."
cd ui

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Warning: node_modules not found. Installing dependencies..."
    npm install
fi

# Start frontend in background
# Note: Vite dev server needs to run in foreground, so we use nohup
nohup npm run dev > ../log/frontend.log 2>&1 &

# Save the frontend process ID
FRONTEND_PID=$!
echo $FRONTEND_PID > ../log/frontend.pid

cd ..

echo "Frontend started in background"
echo "  PID: $FRONTEND_PID"
echo "  Logs: log/frontend.log"
echo ""
echo "======================================"
echo "Both servers are running!"
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:8801 (check log/frontend.log for actual port)"
echo ""
echo "To stop both servers, run: ./stop_all.sh"
echo "Or manually: kill $(cat log/backend.pid) $(cat log/frontend.pid)"
