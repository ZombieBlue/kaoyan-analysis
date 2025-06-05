<template>
  <div class="school-analysis">
    <chart-card title="考研热门学校Top20">
      <div v-if="loading" class="loading-container">
        <el-skeleton :rows="20" animated />
      </div>
      <div v-else>
        <e-charts :options="schoolOptions" height="500px" />
        <div class="data-table">
          <el-table :data="schools" stripe style="width: 100%" height="380px">
            <el-table-column type="index" width="50" />
            <el-table-column prop="school_name" label="学校名称" sortable />
            <el-table-column prop="count" label="申请人数" sortable />
            <el-table-column prop="province" label="所在省份" sortable />
            <el-table-column prop="type" label="学校类型" sortable />
            <el-table-column label="985/211" width="100">
              <template #default="scope">
                <el-tag v-if="scope.row.is_985 === 1" type="success" effect="dark">985</el-tag>
                <el-tag v-if="scope.row.is_211 === 1" type="primary" effect="dark">211</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </chart-card>
    
    <chart-card title="学校申请人数占比分析" v-if="!loading && schools.length > 0">
      <e-charts :options="schoolPieOptions" height="500px" />
    </chart-card>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import ChartCard from '../components/ChartCard.vue'
import ECharts from '../components/ECharts.vue'
import api from '../api'

export default {
  name: 'SchoolAnalysis',
  components: {
    ChartCard,
    ECharts
  },
  setup() {
    const schools = ref([])
    const schoolOptions = ref({})
    const schoolPieOptions = ref({})
    const loading = ref(true)
    
    // 颜色列表
    const colors = [
      '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
      '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#6e7074',
      '#67C23A', '#409EFF', '#E6A23C', '#F56C6C', '#909399',
      '#2c3e50', '#42b983', '#f39c12', '#e74c3c', '#9b59b6'
    ]
    
    // 加载学校数据
    const loadSchoolData = async () => {
      loading.value = true
      try {
        const data = await api.getTopSchools()
        schools.value = data
        
        // 准备图表数据
        const schoolNames = data.map(item => item.school_name)
        const schoolData = data.map(item => ({
          value: item.count,
          itemStyle: {
            color: getSchoolColor(item)
          }
        }))
        
        // 设置图表选项
        schoolOptions.value = {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'shadow'
            },
            formatter: function(params) {
              const item = params[0]
              const index = item.dataIndex
              const school = data[index]
              
              return `
                <div>
                  <p><strong>${school.school_name}</strong></p>
                  <p>申请人数: ${school.count}</p>
                  <p>所在省份: ${school.province}</p>
                  <p>学校类型: ${school.type}</p>
                  <p>是否985: ${school.is_985 ? '是' : '否'}</p>
                  <p>是否211: ${school.is_211 ? '是' : '否'}</p>
                </div>
              `
            }
          },
          grid: {
            left: '5%',
            right: '5%',
            bottom: '10%',
            top: '5%',
            containLabel: true
          },
          xAxis: {
            type: 'category',
            data: schoolNames,
            axisLabel: {
              interval: 0,
              rotate: 45,
              formatter: function(value) {
                if (value.length > 6) {
                  return value.substring(0, 6) + '...'
                }
                return value
              }
            }
          },
          yAxis: {
            type: 'value',
            name: '申请人数'
          },
          series: [
            {
              name: '申请人数',
              type: 'bar',
              data: schoolData,
              itemStyle: {
                borderRadius: 5
              },
              emphasis: {
                focus: 'series'
              },
              label: {
                show: false,
                position: 'top',
                formatter: '{c}'
              }
            }
          ]
        }
        
        // 准备饼图数据
        const schoolPieData = data.slice(0, 10).map((item, index) => ({
          name: item.school_name,
          value: item.count,
          itemStyle: {
            color: colors[index % colors.length]
          }
        }))
        
        // 设置学校饼图选项
        schoolPieOptions.value = {
          tooltip: {
            trigger: 'item',
            formatter: function(params) {
              const school = data.find(s => s.school_name === params.name)
              if (school) {
                return `
                  <div>
                    <p><strong>${school.school_name}</strong></p>
                    <p>申请人数: ${school.count} (${params.percent}%)</p>
                    <p>所在省份: ${school.province}</p>
                    <p>学校类型: ${school.type}</p>
                    <p>是否985: ${school.is_985 ? '是' : '否'}</p>
                    <p>是否211: ${school.is_211 ? '是' : '否'}</p>
                  </div>
                `
              }
              return `${params.name}: ${params.value} (${params.percent}%)`
            }
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
              name: '学校申请人数',
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
              data: schoolPieData
            }
          ]
        }
      } catch (error) {
        console.error('加载学校数据失败:', error)
      } finally {
        loading.value = false
      }
    }
    
    // 根据学校属性获取颜色
    const getSchoolColor = (school) => {
      if (school.is_985 === 1) {
        return '#67C23A'  // 绿色 - 985院校
      } else if (school.is_211 === 1) {
        return '#409EFF'  // 蓝色 - 211院校
      } else {
        return '#E6A23C'  // 橙色 - 普通院校
      }
    }
    
    onMounted(() => {
      loadSchoolData()
    })
    
    return {
      schools,
      schoolOptions,
      schoolPieOptions,
      loading
    }
  }
}
</script>

<style scoped>
.school-analysis {
  padding: 0;
}
.loading-container {
  padding: 20px;
}
.data-table {
  margin-top: 20px;
}
</style> 