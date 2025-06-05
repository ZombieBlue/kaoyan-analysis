<template>
  <div class="dashboard">
    <!-- 统计卡片区域 -->
    <el-row :gutter="20" class="data-overview">
      <el-col :span="8">
        <data-card
          title="考研热门学校"
          :value="statistics.schoolCount + '所'"
          icon="School"
          bg-color="#409EFF"
        />
      </el-col>
      <el-col :span="8">
        <data-card
          title="考研热门专业"
          :value="statistics.majorCount + '个'"
          icon="Reading"
          bg-color="#E6A23C"
        />
      </el-col>
      <el-col :span="8">
        <data-card
          title="考研热门省份"
          :value="statistics.provinceCount + '个'"
          icon="Location"
          bg-color="#F56C6C"
        />
      </el-col>
    </el-row>
    
    <!-- 图表区域 -->
    <el-row :gutter="20" class="chart-row">
      <!-- 热门学校Top10 -->
      <el-col :span="12">
        <chart-card title="考研热门学校Top10">
          <e-charts :options="schoolOptions" height="300px" />
        </chart-card>
      </el-col>
      
      <!-- 热门专业Top10 -->
      <el-col :span="12">
        <chart-card title="考研热门专业Top10">
          <e-charts :options="majorOptions" height="300px" />
        </chart-card>
      </el-col>
    </el-row>
    
    <!-- 省份分布 -->
    <el-row>
      <el-col :span="24">
        <chart-card title="考研热门省份分布">
          <e-charts :options="provinceOptions" height="400px" />
        </chart-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'
import DataCard from '../components/DataCard.vue'
import ChartCard from '../components/ChartCard.vue'
import ECharts from '../components/ECharts.vue'
import api from '../api'

export default {
  name: 'Dashboard',
  components: {
    DataCard,
    ChartCard,
    ECharts
  },
  setup() {
    // 统计数据
    const statistics = reactive({
      schoolCount: 0,
      majorCount: 0,
      provinceCount: 0
    })
    
    // 图表配置
    const schoolOptions = ref({})
    const majorOptions = ref({})
    const provinceOptions = ref({})
    
    // 数据加载状态
    const loading = ref(false)
    
    // 加载所有数据
    const loadAllData = async () => {
      loading.value = true
      try {
        const data = await api.getAllStats()
        
        // 更新统计数据
        statistics.schoolCount = data.top_schools.length
        statistics.majorCount = data.top_majors.length
        statistics.provinceCount = data.top_provinces.length
        
        // 准备学校图表数据
        const schoolData = data.top_schools.slice(0, 10).map(item => item.count)
        const schoolNames = data.top_schools.slice(0, 10).map(item => item.school_name)
        
        schoolOptions.value = {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'shadow'
            }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
          },
          xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
          },
          yAxis: {
            type: 'category',
            data: schoolNames.reverse(),
            axisLabel: {
              interval: 0,
              rotate: 0,
              formatter: function(value) {
                if (value.length > 8) {
                  return value.substring(0, 8) + '...'
                }
                return value
              }
            }
          },
          series: [
            {
              name: '申请人数',
              type: 'bar',
              data: schoolData.reverse(),
              itemStyle: {
                color: '#409EFF'
              }
            }
          ]
        }
        
        // 准备专业图表数据
        const majorData = data.top_majors.slice(0, 10).map(item => item.count)
        const majorNames = data.top_majors.slice(0, 10).map(item => item.name)
        
        majorOptions.value = {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'shadow'
            }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
          },
          xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
          },
          yAxis: {
            type: 'category',
            data: majorNames.reverse(),
            axisLabel: {
              interval: 0,
              rotate: 0,
              formatter: function(value) {
                if (value.length > 10) {
                  return value.substring(0, 10) + '...'
                }
                return value
              }
            }
          },
          series: [
            {
              name: '申请人数',
              type: 'bar',
              data: majorData.reverse(),
              itemStyle: {
                color: '#E6A23C'
              }
            }
          ]
        }
        
        // 准备省份图表数据
        const provinceData = data.top_provinces.map(item => ({
          name: item.province,
          value: item.count
        }))
        
        provinceOptions.value = {
          tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
          },
          legend: {
            orient: 'horizontal',
            bottom: 10,
            data: data.top_provinces.map(item => item.province)
          },
          series: [
            {
              name: '省份分布',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
              },
              label: {
                show: false,
                position: 'center'
              },
              emphasis: {
                label: {
                  show: true,
                  fontSize: '18',
                  fontWeight: 'bold'
                }
              },
              labelLine: {
                show: false
              },
              data: provinceData
            }
          ]
        }
      } catch (error) {
        console.error('加载数据失败:', error)
      } finally {
        loading.value = false
      }
    }
    
    onMounted(() => {
      loadAllData()
    })
    
    return {
      statistics,
      schoolOptions,
      majorOptions,
      provinceOptions,
      loading
    }
  }
}
</script>

<style scoped>
.dashboard {
  padding: 0;
}
.data-overview {
  margin-bottom: 20px;
}
.chart-row {
  margin-bottom: 20px;
}
</style> 