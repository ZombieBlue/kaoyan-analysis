<template>
  <div ref="chart" class="chart-container"></div>
</template>

<script>
import { onMounted, onBeforeUnmount, watch, shallowRef } from 'vue'
import * as echarts from 'echarts'

export default {
  name: 'ECharts',
  props: {
    options: {
      type: Object,
      required: true
    },
    theme: {
      type: String,
      default: ''
    }
  },
  setup(props) {
    const chart = shallowRef(null)
    const chartInstance = shallowRef(null)
    
    // 初始化图表
    const initChart = () => {
      if (chartInstance.value) {
        chartInstance.value.dispose()
      }
      
      chartInstance.value = echarts.init(chart.value, props.theme)
      chartInstance.value.setOption(props.options)
      
      window.addEventListener('resize', resize)
    }
    
    // 调整大小
    const resize = () => {
      if (chartInstance.value) {
        chartInstance.value.resize()
      }
    }
    
    // 组件挂载时初始化图表
    onMounted(() => {
      initChart()
    })
    
    // 组件卸载前释放资源
    onBeforeUnmount(() => {
      if (chartInstance.value) {
        chartInstance.value.dispose()
        chartInstance.value = null
      }
      window.removeEventListener('resize', resize)
    })
    
    // 监听选项变化，更新图表
    watch(
      () => props.options,
      (newOptions) => {
        if (chartInstance.value) {
          chartInstance.value.setOption(newOptions)
        }
      },
      { deep: true }
    )
    
    return {
      chart
    }
  }
}
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
}
</style> 