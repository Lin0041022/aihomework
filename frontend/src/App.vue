<template>
  <div id="app">
    <div class="container">
      <div class="header">
        <h1>🏠 房价数据分析系统</h1>
        <p>智能化房价数据分析与可视化平台</p>
        <button @click="exportAnalysis" class="btn btn-primary export-btn">
          📤 导出分析结果
        </button>
      </div>

      <div class="content">
        <div class="main-grid">
          <!-- 左侧操作区域 -->
          <div class="left-column">
            <!-- 数据加载区域 -->
            <div class="section">
              <h2 class="section-title">
                <span class="section-icon">📊</span>
                数据加载
              </h2>

              <div class="form-group">
               <!-- <label class="form-label">选择历史导入数据</label>
                <select v-model="selectedImportId" class="form-select" style="margin-bottom: 10px;">
                  <option disabled value="">请选择导入记录</option>
                  <option v-for="record in importRecords" :key="record.import_id" :value="record.import_id">
                    {{ record.import_time }} - {{ record.file_path }} ({{ record.row_count }}行)
                  </option>
                </select>
                <button @click="loadHistory" :disabled="!selectedImportId" class="btn btn-primary">
                  📂 加载历史数据
                </button>-->
              </div>

              <div class="form-group">
                <label class="form-label">上传新数据文件</label>
                <div class="file-upload-area"
                     @click="$refs.fileInput.click()"
                     @dragover.prevent="onDragOver"
                     @dragleave.prevent="onDragLeave"
                     @drop.prevent="onDrop"
                     :class="{ 'drag-over': isDragOver }">
                  <div class="upload-icon">📁</div>
                  <div class="upload-text">点击选择文件或拖拽文件到此处</div>
                  <div class="upload-hint">支持 CSV 格式</div>
                  <div v-if="selectedFile" class="selected-file">
                    已选择: {{ selectedFile.name }}
                  </div>
                </div>
                <input ref="fileInput" type="file" accept=".csv" @change="onFileChange" style="display: none;" />
                <button @click="loadNewData" :disabled="!selectedFile" class="btn btn-secondary" style="margin-top: 10px;">
                  ⬆️ 上传新数据
                </button>
              </div>
            </div>

            <!-- 数据处理工作流 -->
            <div class="workflow-container">
              <div class="workflow-step" v-for="step in workflowSteps" :key="step.number">
                <div class="step-number">{{ step.number }}</div>
                <div class="step-title">{{ step.title }}</div>
                <div class="step-description">{{ step.description }}</div>
                <button @click="step.action" :disabled="step.disabled" :class="['btn', step.buttonClass]">
                  {{ step.icon }} {{ step.buttonText }}
                </button>
              </div>
            </div>
          </div>

          <!-- 右侧展示区域 -->
          <div class="right-column">
            <!-- 数据概览 -->
            <div class="data-overview">
              <h3 class="overview-title">
                📋 数据概览
              </h3>
              <div v-if="currentDataOverview" class="overview-grid">
                <div class="overview-item">
                  <div class="overview-label">总记录数</div>
                  <div class="overview-value">{{ currentDataOverview.total_rows }}</div>
                </div>
                <div class="overview-item">
                  <div class="overview-label">列数</div>
                  <div class="overview-value">{{ currentDataOverview.total_columns }}</div>
                </div>
                <div class="overview-item">
                  <div class="overview-label">缺失值</div>
                  <div class="overview-value">{{ currentDataOverview.missing_values }}</div>
                </div>
                <div class="overview-item">
                  <div class="overview-label">均价(万元)</div>
                  <div class="overview-value">{{ currentDataOverview.price_range.mean ? currentDataOverview.price_range.mean.toFixed(2) : '-' }}</div>
                </div>
              </div>
              <div v-else class="placeholder">
                <div class="placeholder-icon">📊</div>
                <div class="placeholder-text">暂无数据概览</div>
                <div class="placeholder-hint">请在左侧加载数据以查看概览信息</div>
              </div>
              <div v-if="currentDataOverview && currentDataOverview.district_distribution" style="margin-top: 20px;">
                <div class="overview-label">涉及区域</div>
                <div style="margin-top: 10px;">
                  <span v-for="district in Object.keys(currentDataOverview.district_distribution)" :key="district"
                        style="display: inline-block; background: #e0f2fe; color: #0c4a6e; padding: 4px 12px; margin: 2px; border-radius: 20px; font-size: 0.85rem;">
                    {{ district }}
                  </span>
                </div>
              </div>
            </div>

            <!-- 可视化图表 -->
            <div class="image-container">
              <!-- 图表类型选择器 -->
              <div v-if="imageUrl || modelEvaluationImages.length > 0" class="chart-selector">
                <div class="chart-tabs">
                  <button
                    v-for="tab in availableChartTabs.filter(tab => tab && tab.key)"
                    :key="tab.key"
                    @click="activeChartTab = tab.key"
                    :class="['chart-tab', { active: activeChartTab === tab.key }]"
                  >
                    {{ tab.icon }} {{ tab.name }}
                  </button>
                </div>
              </div>

              <!-- 基础数据分析图表 -->
              <div v-if="activeChartTab === 'basic' && imageUrl" class="chart-display">
                <img :src="imageUrl" alt="房价分析可视化图" class="chart-image" />
                <div class="chart-info">
                  <h4>📊 基础数据分析</h4>
                  <p>包含价格分布、区域对比、户型分布等基础统计图表</p>
                </div>
              </div>

              <!-- 模型评估图表 -->
              <div v-if="activeChartTab === 'model' && currentModelChart" class="chart-display">
                <img :src="currentModelChart.url" :alt="currentModelChart.name" class="chart-image" />
                <div class="chart-info">
                  <h4>{{ currentModelChart.icon }} {{ currentModelChart.name }}</h4>
                  <p>{{ currentModelChart.description }}</p>
                  <!-- 模型性能数据显示 -->
                  <div v-if="modelPerformanceData && activeChartTab === 'model'" class="performance-metrics">
                    <div class="metrics-grid">
                      <div class="metric-item">
                        <span class="metric-label">R²分数</span>
                        <span class="metric-value">{{ modelPerformanceData.r2_score ? modelPerformanceData.r2_score.toFixed(3) : '-' }}</span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">均方误差</span>
                        <span class="metric-value">{{ modelPerformanceData.mean_squared_error ? modelPerformanceData.mean_squared_error.toFixed(2) : '-' }}</span>
                      </div>
                      <div class="metric-item">
                        <span class="metric-label">平均绝对误差</span>
                        <span class="metric-value">{{ modelPerformanceData.mean_absolute_error ? modelPerformanceData.mean_absolute_error.toFixed(2) : '-' }}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <!-- 模型图表切换器 -->
                <div class="model-chart-selector">
                  <button
                    v-for="chart in modelEvaluationCharts.filter(chart => chart && chart.key)"
                    :key="chart.key"
                    @click="selectModelChart(chart.key)"
                    :class="['model-chart-btn', { active: currentModelChartKey === chart.key }]"
                  >
                    {{ chart.icon }} {{ chart.name }}
                  </button>
                </div>
              </div>

              <!-- 暂无图表时的占位符 -->
              <div v-if="!imageUrl && modelEvaluationImages.length === 0" class="placeholder">
                <div class="placeholder-icon">📈</div>
                <div class="placeholder-text">暂无可视化图表</div>
                <div class="placeholder-hint">请完成数据处理并获取图表</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 状态信息 -->
        <div v-if="statusMessage" class="status-message" :class="getStatusClass()">
          <span>{{ getStatusIcon() }}</span>
          <span>{{ statusMessage }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'

const importRecords = ref([])
const selectedImportId = ref('')
const selectedFile = ref(null)
const statusMessage = ref('')
const currentDataOverview = ref(null)
const imageUrl = ref(null)
const isDragOver = ref(false)
const activeChartTab = ref('basic')
const currentModelChartKey = ref(null)
const currentModelChart = ref(null)
const modelPerformanceData = ref(null)
const modelEvaluationImages = ref([])
const modelEvaluationCharts = ref([])

const workflowSteps = computed(() => [
  { number: 1, title: '数据预处理', description: '清洗和标准化数据，处理缺失值和异常值', action: preprocessData, icon: '🔧', buttonText: '执行预处理', buttonClass: 'btn-primary', disabled: false },
  { number: 2, title: '构建预测模型', description: '使用机器学习算法训练房价预测模型', action: buildModel, icon: '🤖', buttonText: '构建模型', buttonClass: 'btn-primary', disabled: false },
  { number: 3, title: '生成房价分析', description: '基于模型和数据生成房价分析报告', action: generateAnalysis, icon: '📈', buttonText: '生成分析', buttonClass: 'btn-primary', disabled: false },
  { number: 4, title: '可视化分析', description: '生成直观的图表和报告展示分析结果', action: fetchVisualization, icon: '📊', buttonText: '获取图表', buttonClass: 'btn-secondary', disabled: false },
  { number: 5, title: '模型评估', description: '生成模型评估图表 (需先完成前4步)', action: fetchModelEvaluationCharts, icon: '🎯', buttonText: '模型评估', buttonClass: 'btn-info', disabled: false },
  { number: 6, title: '系统调试', description: '检查系统状态和数据完整性', action: checkSystemStatus, icon: '🔍', buttonText: '检查状态', buttonClass: 'btn-warning', disabled: false }
])

const availableChartTabs = computed(() => {
  const tabs = []
  if (imageUrl.value) {
    tabs.push({ key: 'basic', name: '基础分析', icon: '📊' })
  }
  if (modelEvaluationImages.value && modelEvaluationImages.value.length > 0) {
    tabs.push({ key: 'model', name: '模型评估', icon: '🎯' })
  }
  return tabs
})

const initializeModelEvaluationCharts = () => {
  const charts = [
    { key: 'roc_pr', name: 'ROC/PR曲线', icon: '📈', description: '展示模型的真正率vs假正率和精确率vs召回率，用于评估回归性能' },
    { key: 'confusion_matrix', name: '残差分布', icon: '🎭', description: '展示预测残差的分布，直观显示回归误差' }
  ]
  modelEvaluationCharts.value = charts.filter(chart => chart && chart.key && chart.name && chart.icon && chart.description)
}
initializeModelEvaluationCharts()

async function fetchImportRecords() {
  try {
    const res = await axios.get('/records/imports')
    importRecords.value = res.data.success ? res.data.records || [] : []
  } catch (error) {
    alert('获取导入记录失败：' + error.message)
  }
}

function onFileChange(event) {
  selectedFile.value = event.target.files[0] || null
  if (selectedFile.value) {
    console.log('选择文件:', selectedFile.value.name)
  }
}

function onDragOver(event) {
  isDragOver.value = true
}

function onDragLeave(event) {
  isDragOver.value = false
}

function onDrop(event) {
  isDragOver.value = false
  const files = event.dataTransfer.files
  if (files.length > 0) {
    selectedFile.value = files[0]
    console.log('拖拽文件:', selectedFile.value.name)
  }
}

async function loadHistory() {
  if (!selectedImportId.value) return
  try {
    statusMessage.value = '加载历史数据中...'
    const res = await axios.post('/data/load-history', new URLSearchParams({ import_id: selectedImportId.value }), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
    statusMessage.value = res.data.success ? res.data.message || '历史数据加载成功' : '加载历史数据失败：' + (res.data.message || '未知错误')
    currentDataOverview.value = res.data.success ? res.data.data_overview || null : null
  } catch (error) {
    statusMessage.value = '加载历史数据失败: ' + (error.response?.data?.detail || error.message)
  }
}

async function loadNewData() {
  if (!selectedFile.value) {
    statusMessage.value = '请先选择文件'
    return
  }
  try {
    statusMessage.value = '上传新数据中...'
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    const res = await axios.post('/data/load-new', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    if (res.data.success) {
      statusMessage.value = '新数据上传成功'
      currentDataOverview.value = res.data.data_overview || null
      await fetchImportRecords()
    } else {
      statusMessage.value = '上传新数据失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    console.error('上传错误:', error)
    statusMessage.value = '上传新数据失败: ' + (error.response?.data?.detail || error.message)
  }
}

async function preprocessData() {
  try {
    statusMessage.value = '数据预处理中...'
    const res = await axios.post('/data/preprocess')
    statusMessage.value = res.data.success ? '数据预处理完成' : '数据预处理失败：' + (res.data.message || '未知错误')
  } catch (error) {
    statusMessage.value = '数据预处理失败: ' + (error.response?.data?.detail || error.message)
  }
}

async function buildModel() {
  try {
    statusMessage.value = '模型构建中...'
    const res = await axios.post('/model/build')
    statusMessage.value = res.data.success ? '模型构建完成' : '模型构建失败：' + (res.data.message || '未知错误')
  } catch (error) {
    statusMessage.value = '模型构建失败: ' + (error.response?.data?.detail || error.message)
  }
}

async function generateAnalysis() {
  try {
    statusMessage.value = '生成房价分析中...'
    const res = await axios.post('/analysis/generate')
    statusMessage.value = res.data.success ? '房价分析生成完成' : '生成房价分析失败：' + (res.data.message || '未知错误')
  } catch (error) {
    statusMessage.value = '生成房价分析失败: ' + (error.response?.data?.detail || error.message)
  }
}

async function fetchVisualization() {
  try {
    const response = await fetch('/visualizations')
    if (response.ok) {
      imageUrl.value = URL.createObjectURL(await response.blob())
      statusMessage.value = '可视化图表加载成功'
    } else {
      throw new Error('获取图表失败')
    }
  } catch (error) {
    statusMessage.value = '获取可视化图表失败：' + error.message
  }
}

async function fetchModelEvaluationCharts() {
  try {
    statusMessage.value = '正在生成模型评估图表...'
    const response = await fetch('/visualizations/model-evaluation')
    if (response.ok) {
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      modelEvaluationImages.value = [
        {
          key: 'roc_pr',
          name: '模型评估图表',
          icon: '📈',
          description: '模型评估图表',
          url: url
        }
      ]
      currentModelChartKey.value = 'roc_pr'
      currentModelChart.value = modelEvaluationImages.value[0]
      activeChartTab.value = 'model'
      statusMessage.value = '模型评估图表加载成功'
    } else {
      modelEvaluationImages.value = []
      statusMessage.value = '模型评估图表生成失败：图表数据无效'
    }
  } catch (error) {
    statusMessage.value = '获取模型评估图表失败：' + error.message
  }
}

function selectModelChart(chartKey) {
  if (!chartKey) return
  const chart = modelEvaluationImages.value.find(img => img && img.key === chartKey)
  if (chart) {
    currentModelChartKey.value = chartKey
    currentModelChart.value = chart
  }
}

function getStatusClass() {
  return statusMessage.value.includes('失败') || statusMessage.value.includes('错误') ? 'status-error'
    : statusMessage.value.includes('成功') || statusMessage.value.includes('完成') ? 'status-success'
    : 'status-info'
}

function getStatusIcon() {
  return statusMessage.value.includes('失败') || statusMessage.value.includes('错误') ? '❌'
    : statusMessage.value.includes('成功') || statusMessage.value.includes('完成') ? '✅'
    : 'ℹ️'
}

async function checkSystemStatus() {
  try {
    statusMessage.value = '正在检查系统状态...'
    let statusDetails = '系统状态检查:\n'
    statusMessage.value = statusDetails
  } catch (error) {
    statusMessage.value = '检查系统状态失败：' + error.message
  }
}

async function exportAnalysis() {
  try {
    statusMessage.value = '正在导出分析结果...'
    const response = await fetch('/export/analysis')
    if (response.ok) {
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'house_price_analysis.xlsx'
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
      statusMessage.value = '分析结果导出成功'
    } else {
      statusMessage.value = '导出分析结果失败'
    }
  } catch (error) {
    statusMessage.value = '导出分析结果失败：' + error.message
  }
}

onMounted(() => {
  fetchImportRecords()
})
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
  min-height: 100vh;
  color: #22223b;
}
.container {
  max-width: 1200px;
  margin: 32px auto;
  background: rgba(255,255,255,0.98);
  border-radius: 24px;
  box-shadow: 0 8px 40px rgba(60,72,100,0.13);
  overflow: hidden;
  min-height: 80vh;
  display: flex;
  flex-direction: column;
}
.header {
  background: linear-gradient(90deg, #6366f1 0%, #7c3aed 100%);
  color: white;
  padding: 36px 48px 24px 48px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  border-bottom: 2px solid rgba(255,255,255,0.08);
  position: relative;
  overflow: hidden;
}
.header h1 {
  font-size: 2.6rem;
  font-weight: 900;
  margin-bottom: 8px;
  letter-spacing: 1px;
  background: linear-gradient(90deg, #fff, #e0e7ff 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.header p {
  font-size: 1.2rem;
  opacity: 0.92;
  margin-bottom: 12px;
}
.export-btn {
  margin-left: auto;
  margin-top: 10px;
  background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%);
  color: white;
  border: none;
  border-radius: 16px;
  font-size: 1.1rem;
  padding: 12px 32px;
  box-shadow: 0 4px 16px rgba(16,185,129,0.13);
  transition: transform 0.2s, box-shadow 0.2s;
}
.export-btn:hover {
  transform: translateY(-2px) scale(1.04);
  box-shadow: 0 8px 32px rgba(16,185,129,0.18);
}
.content {
  padding: 36px 48px;
  flex: 1;
  display: flex;
  flex-direction: column;
}
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 36px;
  flex: 1;
}
.left-column, .right-column {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.section {
  padding: 28px 24px;
  background: #f4f7fa;
  border-radius: 18px;
  box-shadow: 0 4px 24px rgba(99,102,241,0.07);
  border: 1px solid #e0e7ff;
  transition: box-shadow 0.3s, transform 0.3s;
  position: relative;
  overflow: hidden;
}
.section:hover {
  box-shadow: 0 8px 32px rgba(99,102,241,0.13);
  transform: translateY(-4px) scale(1.01);
}
.section-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #3730a3;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.section-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #6366f1, #7c3aed);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 18px;
  box-shadow: 0 2px 8px rgba(99,102,241,0.18);
}
.form-group {
  margin-bottom: 24px;
}
.form-label {
  display: block;
  font-weight: 600;
  color: #3730a3;
  margin-bottom: 10px;
  font-size: 1.08rem;
}
.form-select {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e0e7ff;
  border-radius: 12px;
  font-size: 1.08rem;
  background: #fff;
  color: #22223b;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.form-select:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px #e0e7ff;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 22px;
  border: none;
  border-radius: 14px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
  position: relative;
  overflow: hidden;
}
.btn-primary {
  background: linear-gradient(90deg, #6366f1 0%, #7c3aed 100%);
  color: white;
  box-shadow: 0 4px 16px rgba(99,102,241,0.13);
}
.btn-primary:hover {
  transform: translateY(-2px) scale(1.03);
  box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
.btn-secondary {
  background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%);
  color: white;
  box-shadow: 0 4px 16px rgba(16,185,129,0.13);
}
.btn-secondary:hover {
  transform: translateY(-2px) scale(1.03);
  box-shadow: 0 8px 32px rgba(16,185,129,0.18);
}
.btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
.workflow-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
.workflow-step {
  background: #fff;
  padding: 18px 14px;
  border-radius: 14px;
  box-shadow: 0 2px 12px rgba(99,102,241,0.07);
  border: 1px solid #e0e7ff;
  transition: box-shadow 0.3s, transform 0.3s;
  position: relative;
  overflow: hidden;
}
.workflow-step:hover {
  box-shadow: 0 8px 24px rgba(99,102,241,0.13);
  transform: translateY(-2px) scale(1.01);
}
.step-number {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #6366f1, #7c3aed);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  margin-bottom: 10px;
  font-size: 1.1rem;
  box-shadow: 0 2px 8px rgba(99,102,241,0.18);
}
.step-title {
  font-size: 1.08rem;
  font-weight: 600;
  color: #3730a3;
  margin-bottom: 6px;
}
.step-description {
  color: #6b7280;
  font-size: 0.95rem;
  margin-bottom: 10px;
  line-height: 1.5;
}
.data-overview {
  background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 100%);
  border: 1px solid #6366f1;
  border-radius: 18px;
  padding: 28px 24px;
  min-height: 260px;
  box-shadow: 0 8px 32px rgba(99,102,241,0.07);
}
.overview-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #3730a3;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.overview-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
}
.overview-item {
  background: white;
  padding: 12px 10px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(99,102,241,0.05);
  border-left: 4px solid #6366f1;
  transition: transform 0.2s;
}
.overview-item:hover {
  transform: translateY(-2px) scale(1.01);
}
.overview-label {
  font-size: 1rem;
  color: #6366f1;
  margin-bottom: 6px;
  font-weight: 600;
}
.overview-value {
  font-size: 1.2rem;
  font-weight: 700;
  color: #22223b;
}
.placeholder {
  text-align: center;
  padding: 36px 0;
  background: white;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(99,102,241,0.05);
  border: 1px solid #e0e7ff;
  min-height: 120px;
  transition: all 0.2s;
}
.placeholder-icon {
  font-size: 2.5rem;
  color: #9ca3af;
  margin-bottom: 10px;
}
.placeholder-text {
  font-size: 1.1rem;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 6px;
}
.placeholder-hint {
  font-size: 0.98rem;
  color: #9ca3af;
}
.status-message {
  margin-top: 18px;
  padding: 10px 18px;
  border-radius: 10px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
  animation: slideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.status-info {
  background: linear-gradient(120deg, #dbeafe 0%, #bfdbfe 100%);
  color: #3730a3;
  border: 1px solid #6366f1;
}
.status-success {
  background: linear-gradient(120deg, #d1fae5 0%, #a7f3d0 100%);
  color: #065f46;
  border: 1px solid #10b981;
}
.status-error {
  background: linear-gradient(120deg, #fee2e2 0%, #fecaca 100%);
  color: #dc2626;
  border: 1px solid #ef4444;
}
.image-container {
  text-align: center;
  min-height: 320px;
  background: white;
  border-radius: 18px;
  padding: 18px 10px;
  box-shadow: 0 8px 32px rgba(99,102,241,0.07);
  transition: all 0.2s;
}
.image-container:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow: 0 12px 40px rgba(99,102,241,0.13);
}
.chart-image {
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(99,102,241,0.08);
  border: 1px solid #e0e7ff;
  transition: all 0.2s;
}
.chart-image:hover {
  transform: scale(1.02);
}
.file-upload-area {
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  padding: 18px;
  text-align: center;
  transition: all 0.2s;
  cursor: pointer;
  background: #f9fafb;
  position: relative;
  overflow: hidden;
}
.file-upload-area:hover {
  border-color: #6366f1;
  background: #eef2ff;
  transform: translateY(-2px);
}
.file-upload-area.drag-over {
  border-color: #6366f1;
  background: #eef2ff;
  transform: translateY(-2px);
}
.selected-file {
  margin-top: 8px;
  padding: 6px 10px;
  background: #10b981;
  color: white;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 500;
}
.upload-icon {
  font-size: 2rem;
  color: #9ca3af;
  margin-bottom: 8px;
  transition: transform 0.2s;
}
.file-upload-area:hover .upload-icon {
  transform: scale(1.08);
}
.upload-text {
  color: #6b7280;
  font-size: 1rem;
  margin-bottom: 6px;
  font-weight: 500;
}
.upload-hint {
  color: #9ca3af;
  font-size: 0.92rem;
}
@media (max-width: 1024px) {
  .main-grid {
    grid-template-columns: 1fr;
    gap: 18px;
  }
  .workflow-container {
    grid-template-columns: 1fr;
  }
  .overview-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 768px) {
  .container {
    margin: 8px;
    border-radius: 14px;
  }
  .header {
    padding: 18px;
    flex-direction: column;
    align-items: flex-start;
  }
  .header h1 {
    font-size: 1.5rem;
  }
  .export-btn {
    margin-top: 10px;
    margin-left: 0;
  }
  .content {
    padding: 14px;
  }
  .workflow-container {
    grid-template-columns: 1fr;
  }
  .overview-grid {
    grid-template-columns: 1fr;
  }
  .image-container {
    min-height: 180px;
    padding: 10px;
  }
  .btn {
    padding: 8px 14px;
    font-size: 0.95rem;
  }
}
@keyframes slideIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>