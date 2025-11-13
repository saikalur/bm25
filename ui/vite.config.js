import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8801,
    strictPort: true, // Fail if port is already in use instead of trying another port
    host: true,
    open: false,
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      'app.bakermatcher.com'
    ]
  }
});

