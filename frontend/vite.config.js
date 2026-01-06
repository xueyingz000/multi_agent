import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.BACKEND_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 8080,
      proxy: {
        '/upload': {
          target: target,
          changeOrigin: true,
          secure: false,
        },
        '/analyze': {
          target: target,
          changeOrigin: true,
          secure: false,
        },
        '/calculate': {
          target: target,
          changeOrigin: true,
          secure: false,
        },
        '/export': {
          target: target,
          changeOrigin: true,
          secure: false,
        },
      }
    },
    preview: {
      host: '0.0.0.0',
      port: 8080,
      allowedHosts: ['multi-agent.zeabur.app']
    }
  }
})