<template>
  <div class="score-analysis">
    <chart-card title="热门Top10学校专业录取分数线预测">
      <div v-if="loading" class="loading-container">
        <el-skeleton :rows="20" animated />
      </div>
      <div v-else>
        <!-- 学校选择器 -->
        <div class="filters">
          <el-select v-model="selectedSchool" placeholder="选择学校" @change="handleSchoolChange">
            <el-option
              v-for="item in topSchools"
              :key="item.school_id"
              :label="item.school_name"
              :value="item.school_id">
            </el-option>
          </el-select>
          
          <el-select v-model="selectedMajor" placeholder="选择专业" @change="updateScoreChart">
            <el-option
              v-for="item in majors"
              :key="item.code"
              :label="item.name"
              :value="item.code">
            </el-option>
          </el-select>
        </div>
        
        <!-- 分数线折线图 -->
        <e-charts :options="scoreChartOptions" height="400px" />
        
        <!-- 分数预测表格 -->
        <div class="data-table">
          <h3>{{ getSelectedSchoolName() }} - {{ getSelectedMajorName() }} 分数线预测</h3>
          <el-table 
            :data="predictionData" 
            stripe 
            style="width: 100%" 
            height="450px"
            :default-sort="{prop: 'year', order: 'descending'}">
            <el-table-column prop="year" label="年份" width="120" sortable fixed>
              <template #default="scope">
                <span :class="{ 'highlight-year': scope.row.year >= 2024 }">{{ scope.row.year }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="score" label="实际分数线" sortable>
              <template #default="scope">
                <span v-if="scope.row.score !== null">{{ scope.row.score }}</span>
                <span v-else class="no-data">--</span>
              </template>
            </el-table-column>
            <el-table-column prop="predicted" label="预测分数线" sortable>
              <template #default="scope">
                <span v-if="scope.row.predicted !== null" :class="getPredictionClass(scope.row)">
                  {{ scope.row.predicted }}
                </span>
                <span v-else class="no-data">--</span>
              </template>
            </el-table-column>
            <el-table-column prop="difference" label="变化情况" sortable>
              <template #default="scope">
                <span v-if="scope.row.difference !== null" :class="getDifferenceClass(scope.row)">
                  {{ scope.row.difference > 0 ? '+' + scope.row.difference : scope.row.difference }}
                </span>
                <span v-else class="no-data">--</span>
              </template>
            </el-table-column>
          </el-table>
          <div class="table-note">注：2024年数据部分为实际值，部分为预测值；2025年和2026年全部为预测值</div>
        </div>
      </div>
    </chart-card>
    
    <!-- 历史分数线趋势图 -->
    <chart-card title="热门学校历史分数线趋势" v-if="!loading && topSchools.length > 0">
      <e-charts :options="trendChartOptions" height="500px" />
    </chart-card>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed } from 'vue'
import ChartCard from '../components/ChartCard.vue'
import ECharts from '../components/ECharts.vue'
import api from '../api'

export default {
  name: 'ScoreAnalysis',
  components: {
    ChartCard,
    ECharts
  },
  setup() {
    const topSchools = ref([])
    const majors = ref([])
    const selectedSchool = ref('')
    const selectedMajor = ref('')
    const scoreChartOptions = ref({})
    const trendChartOptions = ref({})
    const predictionData = ref([])
    const loading = ref(true)
    
    // 颜色列表
    const colors = [
      '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
      '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#6e7074'
    ]
    
    // 加载热门学校数据
    const loadTopSchools = async () => {
      loading.value = true
      try {
        const data = await api.getTopSchools()
        // 只取前10所学校
        topSchools.value = data.slice(0, 10)
        
        if (topSchools.value.length > 0) {
          selectedSchool.value = topSchools.value[0].school_id
          await loadSchoolMajors(selectedSchool.value)
          await generateHistoricalTrend()
        }
      } catch (error) {
        console.error('加载热门学校数据失败:', error)
      } finally {
        loading.value = false
      }
    }
    
    // 加载学校专业
    const loadSchoolMajors = async (schoolId) => {
      try {
        // 在实际应用中，这里应该调用API获取学校的专业列表
        // 由于我们没有实际的API，这里使用模拟数据
        const mockMajors = [
          { code: '085211', name: '计算机技术' },
          { code: '085400', name: '电子信息' },
          { code: '085300', name: '控制工程' },
          { code: '025100', name: '金融学' },
          { code: '030100', name: '法学' },
          { code: '020200', name: '应用经济学' },
          { code: '040100', name: '教育学' },
          { code: '050100', name: '中国语言文学' }
        ]
        
        majors.value = mockMajors
        
        // 默认选择第一个专业
        if (majors.value.length > 0) {
          selectedMajor.value = majors.value[0].code
          updateScoreChart()
        }
      } catch (error) {
        console.error('加载学校专业数据失败:', error)
      }
    }
    
    // 处理学校选择变化
    const handleSchoolChange = async () => {
      await loadSchoolMajors(selectedSchool.value)
    }
    
    // 生成分数线预测数据
    const generateScorePrediction = (schoolId, majorCode) => {
      // 模拟历史分数线数据
      const historicalYears = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
      const schoolIndex = topSchools.value.findIndex(s => s.school_id === schoolId)
      const majorIndex = majors.value.findIndex(m => m.code === majorCode)
      
      // 根据学校和专业的索引生成基础分数
      const baseScore = 340 + schoolIndex * 5 + majorIndex * 3
      
      // 生成历史分数和预测分数
      const scoreData = historicalYears.map((year, index) => {
        // 历史分数有随机波动
        const randomFactor = Math.floor(Math.random() * 10) - 5
        // 预测分数有上升趋势
        const trendFactor = index * 2
        
        // 2024年及以前的年份显示历史数据，2024年额外显示预测
        let score = null;
        let predicted = null;
        let difference = null;
        
        if (year <= 2024) {
          score = baseScore + trendFactor + randomFactor;
        }
        
        // 2023年和2024年生成预测分数
        if (year === 2023) {
          predicted = baseScore + trendFactor + Math.floor(Math.random() * 8) - 2;
          difference = predicted - score;
        }
        
        if (year === 2024) {
          predicted = baseScore + trendFactor + Math.floor(Math.random() * 8);
          difference = predicted - score;
        }
        
        return {
          year,
          score,
          predicted,
          difference
        }
      })
      
      // 为2025年生成预测分数
      const prediction2025 = {
        year: 2025,
        score: null,
        predicted: baseScore + historicalYears.length * 2 + Math.floor(Math.random() * 10) - 3,
        difference: null
      }
      
      // 为2026年生成预测分数
      const prediction2026 = {
        year: 2026,
        score: null,
        predicted: baseScore + (historicalYears.length + 1) * 2 + Math.floor(Math.random() * 10) - 3,
        difference: null
      }
      
      scoreData.push(prediction2025)
      scoreData.push(prediction2026)
      return scoreData
    }
    
    // 更新分数线图表
    const updateScoreChart = () => {
      // 获取预测数据
      const data = generateScorePrediction(selectedSchool.value, selectedMajor.value)
      predictionData.value = data
      
      // 提取数据系列
      const years = data.map(item => item.year)
      const actualScores = data.map(item => item.score)
      const predictedScores = data.map(item => item.predicted)
      
      // 设置折线图选项
      scoreChartOptions.value = {
        title: {
          text: `${getSelectedSchoolName()} - ${getSelectedMajorName()} 分数线趋势与预测`,
          left: 'center'
        },
        tooltip: {
          trigger: 'axis',
          formatter: function(params) {
            let result = `${params[0].axisValue}年<br/>`;
            params.forEach(param => {
              // 只有当值存在时才显示
              if (param.value !== null) {
                const color = param.color;
                const marker = `<span style="display:inline-block;margin-right:4px;border-radius:10px;width:10px;height:10px;background-color:${color};"></span>`;
                result += `${marker}${param.seriesName}: ${param.value}<br/>`;
              }
            });
            return result;
          }
        },
        legend: {
          data: ['实际分数线', '预测分数线'],
          top: 30
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: 80,
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: years,
          axisLabel: {
            formatter: '{value}'
          }
        },
        yAxis: {
          type: 'value',
          name: '分数',
          min: function(value) {
            return Math.floor(value.min - 10);
          }
        },
        series: [
          {
            name: '实际分数线',
            type: 'line',
            data: actualScores,
            itemStyle: {
              color: '#409EFF'
            },
            lineStyle: {
              width: 3
            },
            symbol: 'circle',
            symbolSize: 8,
            // 为null值设置连接样式
            connectNulls: false
          },
          {
            name: '预测分数线',
            type: 'line',
            data: predictedScores,
            itemStyle: {
              color: '#E6A23C'
            },
            lineStyle: {
              width: 3,
              type: 'dashed'
            },
            symbol: 'rect',
            symbolSize: 8,
            connectNulls: false,
            markPoint: {
              data: [
                { type: 'max', name: '最高分' }
              ]
            },
            markLine: {
              data: [
                { type: 'average', name: '平均分' }
              ]
            }
          }
        ]
      }
    }
    
    // 生成历史分数线趋势图
    const generateHistoricalTrend = async () => {
      // 获取前5所学校
      const schoolsToShow = topSchools.value.slice(0, 5)
      
      // 年份数据
      const years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
      
      // 准备系列数据
      const series = schoolsToShow.map((school, index) => {
        // 为每所学校生成分数线趋势
        const baseScore = 350 + index * 10
        const scores = years.map((year, i) => baseScore + i * 2 + Math.floor(Math.random() * 10) - 5)
        
        // 为2025年创建预测数据
        const prediction2025 = baseScore + years.length * 2 + Math.floor(Math.random() * 10) - 3;
        
        return {
          name: school.school_name,
          type: 'line',
          data: [...scores, null, null], // 添加空值作为2025年和2026年的占位符
          itemStyle: {
            color: colors[index % colors.length]
          },
          lineStyle: {
            width: 2
          },
          symbol: 'circle',
          symbolSize: 6
        }
      })
      
      // 为2025年和2026年生成预测数据系列
      const predictions2025 = schoolsToShow.map((school, index) => {
        const baseScore = 350 + index * 10;
        return baseScore + years.length * 2 + Math.floor(Math.random() * 10) - 3;
      });
      
      const predictions2026 = schoolsToShow.map((school, index) => {
        const baseScore = 350 + index * 10;
        return baseScore + (years.length + 1) * 2 + Math.floor(Math.random() * 10) - 3;
      });
      
      // 为每所学校创建2025年预测数据系列
      const prediction2025Series = schoolsToShow.map((school, index) => {
        return {
          name: `${school.school_name} 2025预测`,
          type: 'line',
          data: [...Array(years.length).fill(null), predictions2025[index], null],
          itemStyle: {
            color: colors[index % colors.length]
          },
          lineStyle: {
            width: 2,
            type: 'dashed'
          },
          symbol: 'rect',
          symbolSize: 8,
          showSymbol: true
        };
      });
      
      // 为每所学校创建2026年预测数据系列
      const prediction2026Series = schoolsToShow.map((school, index) => {
        return {
          name: `${school.school_name} 2026预测`,
          type: 'line',
          data: [...Array(years.length + 1).fill(null), predictions2026[index]],
          itemStyle: {
            color: colors[index % colors.length]
          },
          lineStyle: {
            width: 2,
            type: 'dashed'
          },
          symbol: 'diamond',
          symbolSize: 8,
          showSymbol: true
        };
      });
      
      // 设置趋势图选项
      trendChartOptions.value = {
        title: {
          text: '热门学校录取分数走势及预测',
          subtext: '虚线与特殊标记为预测数据',
          left: 'center',
          top: 10,
          textStyle: {
            fontSize: 20,
            fontWeight: 'bold'
          },
          subtextStyle: {
            fontSize: 14,
            color: '#666'
          }
        },
        tooltip: {
          trigger: 'axis',
          formatter: function(params) {
            let result = `${params[0].axisValue}年<br/>`;
            params.forEach(param => {
              if (param.value !== null && param.value !== undefined) {
                const color = param.color;
                const marker = `<span style="display:inline-block;margin-right:4px;border-radius:10px;width:10px;height:10px;background-color:${color};"></span>`;
                const name = param.seriesName.includes('2025预测') ? 
                  param.seriesName.replace(' 2025预测', '') + '(预测)' : 
                  param.seriesName;
                result += `${marker}${name}: ${param.value}<br/>`;
              }
            });
            return result;
          }
        },
        legend: {
          data: schoolsToShow.map(s => s.school_name),
          formatter: function(name) {
            return name.length > 8 ? name.substring(0, 8) + '...' : name
          },
          top: 70,
          type: 'scroll',
          pageIconSize: 12,
          pageButtonItemGap: 5,
          pageButtonGap: 5,
          pageIconColor: '#666',
          pageIconInactiveColor: '#aaa',
          pageTextStyle: {
            color: '#666'
          }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          top: 150,
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: [...years, 2025, 2026],
          axisLabel: {
            formatter: '{value}'
          }
        },
        yAxis: {
          type: 'value',
          name: '分数',
          min: function(value) {
            return Math.floor(value.min - 15);
          },
          splitLine: {
            lineStyle: {
              type: 'dashed'
            }
          }
        },
        series: [
          ...series,
          ...prediction2025Series,
          ...prediction2026Series
        ]
      }
    }
    
    // 获取选中学校的名称
    const getSelectedSchoolName = () => {
      const school = topSchools.value.find(s => s.school_id === selectedSchool.value)
      return school ? school.school_name : ''
    }
    
    // 获取选中专业的名称
    const getSelectedMajorName = () => {
      const major = majors.value.find(m => m.code === selectedMajor.value)
      return major ? major.name : ''
    }
    
    // 获取预测分数的样式类
    const getPredictionClass = (row) => {
      if (row.difference === null) return ''
      return row.difference > 0 ? 'score-up' : row.difference < 0 ? 'score-down' : ''
    }
    
    // 获取差值的样式类
    const getDifferenceClass = (row) => {
      if (row.difference === null) return ''
      return row.difference > 0 ? 'score-up' : row.difference < 0 ? 'score-down' : ''
    }
    
    onMounted(() => {
      loadTopSchools()
    })
    
    return {
      topSchools,
      majors,
      selectedSchool,
      selectedMajor,
      scoreChartOptions,
      trendChartOptions,
      predictionData,
      loading,
      handleSchoolChange,
      updateScoreChart,
      getSelectedSchoolName,
      getSelectedMajorName,
      getPredictionClass,
      getDifferenceClass
    }
  }
}
</script>

<style scoped>
.score-analysis {
  padding: 0;
}

.loading-container {
  padding: 20px;
}

.filters {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.filters .el-select {
  width: 240px;
}

.data-table {
  margin-top: 20px;
  margin-bottom: 20px;
}

.data-table h3 {
  margin-bottom: 15px;
  text-align: center;
  color: #303133;
  font-size: 18px;
}

.table-note {
  margin-top: 10px;
  color: #909399;
  font-size: 12px;
  text-align: center;
}

.score-up {
  color: #67C23A;
  font-weight: bold;
}

.score-down {
  color: #F56C6C;
  font-weight: bold;
}

.no-data {
  color: #909399;
}

.highlight-year {
  color: #E6A23C;
  font-weight: bold;
  background-color: rgba(255, 236, 184, 0.2);
  padding: 2px 6px;
  border-radius: 4px;
}

/* 添加媒体查询，确保在不同屏幕尺寸下有良好的显示效果 */
@media screen and (max-width: 768px) {
  .filters .el-select {
    width: 100%;
  }
  
  .filters {
    flex-direction: column;
  }
}
</style> 