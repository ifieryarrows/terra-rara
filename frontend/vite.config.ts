import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'https://ifieryarrows-copper-mind.hf.space',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    manifest: true,
    outDir: 'dist',
    sourcemap: false,
  },
})
