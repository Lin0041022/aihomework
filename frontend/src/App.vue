<template>
  <div id="app" style="padding: 20px; max-width: 700px; margin: auto;">
    <h1>学业预警系统</h1>

    <!-- 导入记录选择 -->
    <div style="margin-bottom: 20px;">
      <label for="importSelect"><strong>选择历史导入数据：</strong></label>
      <select id="importSelect" v-model="selectedImportId" style="width: 100%; padding: 6px;">
        <option disabled value="">请选择导入记录</option>
        <option v-for="record in importRecords" :key="record.import_id" :value="record.import_id">
          {{ record.import_time }} - {{ record.file_path }} ({{ record.row_count }}行)
        </option>
      </select>
      <button @click="loadHistory" :disabled="!selectedImportId" style="margin-top: 10px;">加载历史数据</button>
    </div>

    <!-- 上传新数据 -->
    <div style="margin-bottom: 20px;">
      <label><strong>上传新数据文件：</strong></label><br />
      <input type="file" @change="onFileChange" />
      <button @click="loadNewData" :disabled="!selectedFile" style="margin-left: 10px;">加载新数据</button>
    </div>

    <!-- 执行数据预处理 -->
    <div style="margin-bottom: 20px;">
      <button @click="preprocessData">执行数据预处理</button>
    </div>

    <!-- 构建预测模型 -->
    <div style="margin-bottom: 20px;">
      <button @click="buildModel">构建预测模型</button>
    </div>

    <!-- 生成学业预警 -->
    <div style="margin-bottom: 20px;">
      <button @click="generateWarnings">生成学业预警</button>
    </div>

    <!-- 获取可视化图表 -->
    <button @click="fetchVisualization">获取可视化图表</button>
    <div v-if="imageUrl" style="margin-top:20px;">
      <img :src="imageUrl" alt="学业预警可视化图" style="max-width:100%; height:auto;" />
    </div>

    <!-- 数据概览 -->
    <div v-if="currentDataOverview" style="margin-top: 30px; padding: 10px; background: #eef9ff; border-radius: 6px;">
      <h3>数据概览</h3>
      <p>总记录数：{{ currentDataOverview.total_records }}</p>
      <p>列数：{{ currentDataOverview.columns_count }}</p>
      <p>缺失值：{{ currentDataOverview.missing_values }}</p>
      <p>平均分：{{ currentDataOverview.average_score.toFixed(2) }}</p>
      <p>涉及院系：{{ currentDataOverview.departments.join(', ') }}</p>
    </div>

    <!-- 状态显示 -->
    <div v-if="statusMessage" style="margin-top: 30px; padding: 10px; background: #f0f0f0; border-radius: 4px;">
      <strong>状态:</strong> {{ statusMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const importRecords = ref([])
const selectedImportId = ref('')
const selectedFile = ref(null)
const statusMessage = ref('')
const currentDataOverview = ref(null)
const imageUrl = ref(null)

// 获取导入记录列表
async function fetchImportRecords() {
  try {
    const res = await axios.get('/records/imports')
    if (res.data.success) {
      importRecords.value = res.data.records || []
    } else {
      alert('获取导入记录失败：' + (res.data.message || '未知错误'))
    }
  } catch (error) {
    alert('获取导入记录失败：' + error.message)
  }
}

// 选择文件后触发
function onFileChange(event) {
  const files = event.target.files
  if (files.length > 0) {
    selectedFile.value = files[0]
  } else {
    selectedFile.value = null
  }
}

// 加载历史数据
async function loadHistory() {
  if (!selectedImportId.value) return
  try {
    statusMessage.value = '加载历史数据中...'
    const formData = new URLSearchParams()
    formData.append('import_id', selectedImportId.value)
    const res = await axios.post('/data/load-history', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })

    if (res.data.success) {
      statusMessage.value = res.data.message || '历史数据加载成功'
      currentDataOverview.value = res.data.data_overview || null
    } else {
      statusMessage.value = '加载历史数据失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    statusMessage.value = '加载历史数据失败: ' + (error.response?.data?.detail || error.message)
  }
}

// 加载新数据
async function loadNewData() {
  if (!selectedFile.value) return
  try {
    statusMessage.value = '上传新数据中...'
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    const res = await axios.post('/data/load-new', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    if (res.data.success) {
      statusMessage.value = '新数据上传成功'
    } else {
      statusMessage.value = '上传新数据失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    statusMessage.value = '上传新数据失败: ' + (error.response?.data?.detail || error.message)
  }
}

// 执行数据预处理
async function preprocessData() {
  try {
    statusMessage.value = '数据预处理中...'
    const res = await axios.post('/data/preprocess')
    if (res.data.success) {
      statusMessage.value = '数据预处理完成'
    } else {
      statusMessage.value = '数据预处理失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    statusMessage.value = '数据预处理失败: ' + (error.response?.data?.detail || error.message)
  }
}

// 构建预测模型
async function buildModel() {
  try {
    statusMessage.value = '模型构建中...'
    const res = await axios.post('/model/build')
    if (res.data.success) {
      statusMessage.value = '模型构建完成'
    } else {
      statusMessage.value = '模型构建失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    statusMessage.value = '模型构建失败: ' + (error.response?.data?.detail || error.message)
  }
}

// 生成学业预警
async function generateWarnings() {
  try {
    statusMessage.value = '生成学业预警中...'
    const res = await axios.post('/warnings/generate')
    if (res.data.success) {
      statusMessage.value = '学业预警生成完成'
    } else {
      statusMessage.value = '生成学业预警失败：' + (res.data.message || '未知错误')
    }
  } catch (error) {
    statusMessage.value = '生成学业预警失败: ' + (error.response?.data?.detail || error.message)
  }
}

// 获取可视化图表
async function fetchVisualization() {
  try {
    const response = await fetch('/visualizations')
    if (!response.ok) throw new Error('获取图表失败')

    const blob = await response.blob()
    imageUrl.value = URL.createObjectURL(blob)
    statusMessage.value = '可视化图表加载成功'
  } catch (error) {
    statusMessage.value = '获取可视化图表失败：' + error.message
  }
}

onMounted(() => {
  fetchImportRecords()
})
</script>
