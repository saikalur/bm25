#!/bin/bash

# Script to stop both Flask backend server & Frontend

cd "$(dirname "$0")"

echo "Stopping servers..."

# Stop backend
if [ -f log/backend.pid ]; then
    BACKEND_PID=$(cat log/backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID
        echo "✓ Backend server stopped (PID: $BACKEND_PID)"
        rm log/backend.pid
    else
        echo "✗ Backend server was not running"
        rm log/backend.pid
    fi
else
    echo "✗ No backend PID file found"
    # Try to find and kill process on port 5000
    PORT_PID=$(lsof -ti:5000 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        kill $PORT_PID
        echo "✓ Killed process on port 5000 (PID: $PORT_PID)"
    fi
fi

# Stop frontend
if [ -f log/frontend.pid ]; then
    FRONTEND_PID=$(cat log/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID
        echo "✓ Frontend server stopped (PID: $FRONTEND_PID)"
        rm log/frontend.pid
    else
        echo "✗ Frontend server was not running"
        rm log/frontend.pid
    fi
else
    echo "✗ No frontend PID file found"
    # Try to find and kill Vite process
    VITE_PID=$(lsof -ti:5173 2>/dev/null)
    if [ ! -z "$VITE_PID" ]; then
        kill $VITE_PID
        echo "✓ Killed process on port 5173 (PID: $VITE_PID)"
    fi
fi

echo ""
echo "All servers stopped."

