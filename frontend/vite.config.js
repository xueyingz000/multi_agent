import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // 加载环境变量
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react()],
    // 将 BACKEND_URL 暴露给前端代码
    define: {
      'process.env.BACKEND_URL': JSON.stringify(env.BACKEND_URL)
    },
    server: {
      host: '0.0.0.0',
      port: 8080,
    },
    preview: {
      host: '0.0.0.0',
      port: 8080,
      allowedHosts: ['multi-agent.zeabur.app']
    }
  }
})