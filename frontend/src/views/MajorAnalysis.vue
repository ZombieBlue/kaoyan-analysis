<template>
  <div class="major-analysis">
    <chart-card title="考研热门专业Top20">
      <div v-if="loading" class="loading-container">
        <el-skeleton :rows="20" animated />
      </div>
      <div v-else>
        <e-charts :options="majorOptions" height="500px" />
        <div class="data-table">
          <el-table :data="majors" stripe style="width: 100%" height="380px">
            <el-table-column type="index" width="50" />
            <el-table-column prop="code" label="专业代码" width="120" sortable />
            <el-table-column prop="name" label="专业名称" sortable />
            <el-table-column prop="count" label="申请人数" sortable />
            <el-table-column prop="category" label="专业类别" sortable />
          </el-table>
        </div>
      </div>
    </chart-card>
    
    <chart-card title="专业占比分析" v-if="!loading && majors.length > 0">
      <e-charts :options="majorPieOptions" height="400px" />
    </chart-card>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed } from 'vue'
import ChartCard from '../components/ChartCard.vue'
import ECharts from '../components/ECharts.vue'
import api from '../api'

export default {
  name: 'MajorAnalysis',
  components: {
    ChartCard,
    ECharts
  },
  setup() {
    const majors = ref([])
    const majorOptions = ref({})
    const majorPieOptions = ref({})
    const categoryData = ref([])
    const loading = ref(true)
    
    // 颜色列表
    const colors = [
      '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
      '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#6e7074'
    ]
    
    // 加载专业数据
    const loadMajorData = async () => {
      loading.value = true
      try {
        const data = await api.getTopMajors()
        majors.value = data
        
        // 准备图表数据
        const majorNames = data.map(item => item.name)
        const majorData = data.map((item, index) => ({
          value: item.count,
          itemStyle: {
            color: colors[index % colors.length]
          }
        }))
        
        // 专业类别统计
        const categoryMap = {}
        data.forEach(item => {
          const category = item.category || '未分类'
          if (categoryMap[category]) {
            categoryMap[category] += item.count
          } else {
            categoryMap[category] = item.count
          }
        })
        
        // 专业占比数据
        const majorPieData = data.slice(0, 10).map((item, index) => ({
          name: item.name,
          value: item.count,
          itemStyle: {
            color: colors[index % colors.length]
          }
        }))
        
        // 转换为图表数据
        categoryData.value = Object.keys(categoryMap).map((key, index) => ({
          name: key,
          value: categoryMap[key],
          itemStyle: {
            color: colors[index % colors.length]
          }
        }))
        
        // 设置专业图表选项
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
            bottom: '10%',
            top: '3%',
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
                if (value.length > 15) {
                  return value.substring(0, 15) + '...'
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
                borderRadius: 5
              }
            }
          ]
        }
        
        // 设置专业饼图选项
        majorPieOptions.value = {
          tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
          },
          legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            formatter: function(name) {
              return name.length > 10 ? name.substring(0, 10) + '...' : name
            }
          },
          series: [
            {
              name: '专业申请人数',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
              },
              label: {
                show: false
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
              data: majorPieData
            }
          ]
        }
      } catch (error) {
        console.error('加载专业数据失败:', error)
      } finally {
        loading.value = false
      }
    }
    
    onMounted(() => {
      loadMajorData()
    })
    
    return {
      majors,
      majorOptions,
      majorPieOptions,
      categoryData,
      loading
    }
  }
}
</script>

<style scoped>
.major-analysis {
  padding: 0;
}
.loading-container {
  padding: 20px;
}
.data-table {
  margin-top: 20px;
}
</style> 