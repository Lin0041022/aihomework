import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/records': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/data': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/model': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/warnings': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/export': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/student': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/visualizations': {
         target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/analysis': {
         target: 'http://127.0.0.1:8000',
         changeOrigin: true,
      },
    }
  }
})
