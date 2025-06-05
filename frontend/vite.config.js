import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/records': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/data': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/model': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/warnings': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/export': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/student': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/visualizations': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})
