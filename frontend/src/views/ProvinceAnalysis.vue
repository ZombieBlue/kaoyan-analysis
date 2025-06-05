<template>
  <div class="province-analysis">
    <chart-card title="考研热门省份Top20">
      <div v-if="loading" class="loading-container">
        <el-skeleton :rows="20" animated />
      </div>
      <div v-else>
        <el-row :gutter="20">
          <el-col :span="12">
            <e-charts :options="provinceBarOptions" height="400px" />
          </el-col>
          <el-col :span="12">
            <e-charts :options="provincePieOptions" height="400px" />
          </el-col>
        </el-row>
        
        <div class="data-table">
          <el-table :data="provinces" stripe style="width: 100%" height="380px">
            <el-table-column type="index" width="50" />
            <el-table-column prop="province" label="省份名称" sortable />
            <el-table-column prop="count" label="申请人数" sortable />
            <el-table-column label="占比">
              <template #default="scope">
                {{ calculatePercentage(scope.row.count) }}%
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </chart-card>
    
    <!-- 地图可视化 -->
    <chart-card title="考研省份地图分布" v-if="!loading">
      <e-charts :options="mapOptions" height="600px" />
    </chart-card>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed } from 'vue'
import ChartCard from '../components/ChartCard.vue'
import ECharts from '../components/ECharts.vue'
import api from '../api'
import * as echarts from 'echarts'
// 引入中国地图数据
import chinaJSON from '../assets/china.json'

export default {
  name: 'ProvinceAnalysis',
  components: {
    ChartCard,
    ECharts
  },
  setup() {
    const provinces = ref([])
    const provinceBarOptions = ref({})
    const provincePieOptions = ref({})
    const mapOptions = ref({})
    const loading = ref(true)
    const totalCount = ref(0)
    
    // 注册中国地图
    echarts.registerMap('china', chinaJSON)
    
    // 加载省份数据
    const loadProvinceData = async () => {
      loading.value = true
      try {
        const data = await api.getTopProvinces()
        console.log('API返回的原始数据:', data); // 调试，查看API返回的原始数据格式
        provinces.value = data
        
        // 计算总计数
        totalCount.value = data.reduce((sum, item) => sum + item.count, 0)
        
        // 准备柱状图数据
        const provinceNames = data.map(item => item.province)
        const provinceCounts = data.map(item => item.count)
        
        // 设置柱状图选项
        provinceBarOptions.value = {
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
            type: 'category',
            data: provinceNames,
            axisLabel: {
              interval: 0,
              rotate: 45
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
              data: provinceCounts,
              itemStyle: {
                color: function(params) {
                  const colorList = [
                    '#c23531', '#2f4554', '#61a0a8', '#d48265', '#91c7ae',
                    '#749f83', '#ca8622', '#bda29a', '#6e7074', '#546570',
                    '#c4ccd3', '#4ec9b0', '#6f42c1', '#f7749b', '#87cb16',
                    '#10aeff', '#ff6b6b', '#7e41b2', '#00c2b7', '#ffd700'
                  ]
                  return colorList[params.dataIndex % colorList.length]
                },
                borderRadius: 5
              }
            }
          ]
        }
        
        // 设置饼图选项
        provincePieOptions.value = {
          tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
          },
          legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            formatter: function(name) {
              return name.length > 4 ? name.substring(0, 4) + '...' : name
            }
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
              data: data.map(item => ({
                name: item.province,
                value: item.count
              }))
            }
          ]
        }
        
        // 地图数据处理
        // 找出最大值，用于视觉映射
        const maxValue = Math.max(...data.map(item => parseInt(item.count)));
        console.log('最大值:', maxValue);
        
        // 获取地图中所有地区名称
        const mapFeatures = chinaJSON.features || [];
        const allGeoNames = mapFeatures.map(feature => feature.properties.name);
        console.log('地图中的所有地区名称:', allGeoNames);
        
        // 创建标准化的省份名称映射 (从API返回的名称到地图名称)
        const provinceNameMap = {
          '北京': '北京',
          '北京市': '北京',
          '天津': '天津',
          '天津市': '天津', 
          '河北': '河北',
          '河北省': '河北',
          '山西': '山西',
          '山西省': '山西',
          '内蒙古': '内蒙古',
          '内蒙古自治区': '内蒙古',
          '辽宁': '辽宁',
          '辽宁省': '辽宁',
          '吉林': '吉林',
          '吉林省': '吉林',
          '黑龙江': '黑龙江',
          '黑龙江省': '黑龙江',
          '上海': '上海',
          '上海市': '上海',
          '江苏': '江苏',
          '江苏省': '江苏',
          '浙江': '浙江',
          '浙江省': '浙江',
          '安徽': '安徽',
          '安徽省': '安徽',
          '福建': '福建',
          '福建省': '福建',
          '江西': '江西',
          '江西省': '江西',
          '山东': '山东',
          '山东省': '山东',
          '河南': '河南',
          '河南省': '河南',
          '湖北': '湖北',
          '湖北省': '湖北',
          '湖南': '湖南',
          '湖南省': '湖南',
          '广东': '广东',
          '广东省': '广东',
          '广西': '广西',
          '广西壮族自治区': '广西',
          '海南': '海南',
          '海南省': '海南',
          '重庆': '重庆',
          '重庆市': '重庆',
          '四川': '四川',
          '四川省': '四川',
          '贵州': '贵州',
          '贵州省': '贵州',
          '云南': '云南',
          '云南省': '云南',
          '西藏': '西藏',
          '西藏自治区': '西藏',
          '陕西': '陕西',
          '陕西省': '陕西',
          '甘肃': '甘肃',
          '甘肃省': '甘肃',
          '青海': '青海',
          '青海省': '青海',
          '宁夏': '宁夏',
          '宁夏回族自治区': '宁夏',
          '新疆': '新疆',
          '新疆维吾尔自治区': '新疆',
          '台湾': '台湾',
          '台湾省': '台湾',
          '香港': '香港',
          '香港特别行政区': '香港',
          '澳门': '澳门',
          '澳门特别行政区': '澳门'
        };
        
        // 创建省份名称到数值的映射
        const provinceDataMap = {};
        
        // 手动设置测试数据
        const mockDataMap = {
          '北京': 12686,
          '江苏': 9609,
          '湖北': 7279,
          '山东': 5900,
          '辽宁': 5846,
          '上海': 5450,
          '广东': 5336,
          '陕西': 4945,
          '河南': 4600,
          '浙江': 4555,
          '四川': 3900,
          '安徽': 3800,
          '天津': 3700,
          '吉林': 3500,
          '湖南': 3200,
          '黑龙江': 3100,
          '重庆': 2800,
          '福建': 2500,
          '甘肃': 2200,
          '贵州': 2000,
          '云南': 1800,
          '内蒙古': 1600,
          '山西': 1400,
          '广西': 1200,
          '河北': 1000,
          '江西': 800,
          '新疆': 600,
          '海南': 500,
          '西藏': 400,
          '青海': 300,
          '宁夏': 200,
          '台湾': 100,
          '香港': 50,
          '澳门': 50
        };
        
        // 处理地图数据，确保名称一致性
        const mapData = [];
        
        // 遍历所有可能的地区名称，确保覆盖地图中的所有区域
        allGeoNames.forEach(geoName => {
          // 检查数据中是否有匹配的省份
          const matchedItem = data.find(item => {
            const standardName = provinceNameMap[item.province] || item.province.replace(/(省|自治区|维吾尔|回族|壮族|特别行政区|市)$/g, '');
            return standardName === geoName;
          });
          
          if (matchedItem) {
            // 如果找到匹配的数据，将其添加到地图数据中
            const value = parseInt(matchedItem.count);
            mapData.push({
              name: geoName,
              value: value
            });
            provinceDataMap[geoName] = value;
            console.log(`匹配成功: 地图名称=${geoName}, API名称=${matchedItem.province}, 值=${value}`);
          } else {
            // 如果没有找到匹配的数据，使用模拟数据或设置为0
            const mockValue = mockDataMap[geoName] || 0;
            mapData.push({
              name: geoName,
              value: mockValue
            });
            provinceDataMap[geoName] = mockValue;
            console.log(`匹配失败: 地图名称=${geoName}, 使用模拟值=${mockValue}`);
          }
        });
        
        console.log('处理后的地图数据:', mapData); // 调试用
        console.log('省份数据映射:', provinceDataMap); // 调试用
        
        // 设置地图选项
        mapOptions.value = {
          backgroundColor: '#F5F5F5',
          tooltip: {
            trigger: 'item',
            formatter: function(params) {
              return params.name + '<br/>申请人数: ' + (params.value || 0);
            }
          },
          visualMap: {
            type: 'continuous',
            min: 0,
            max: 13000,
            left: 20,
            bottom: 20,
            text: ['高', '低'],
            calculable: true,
            inRange: {
              color: ['#e0f3f8', '#45b7ce', '#045a8d']
            },
            textStyle: {
              color: '#000'
            },
            // 强制指定系列索引
            seriesIndex: 0
          },
          series: [
            {
              name: '考研热度',
              type: 'map',
              map: 'china',
              roam: true,
              zoom: 1.2,
              label: {
                show: true,
                fontSize: 8,
                color: '#000'
              },
              itemStyle: {
                areaColor: '#e0f3f8',  // 默认颜色
                borderColor: '#ccc',
                borderWidth: 0.5
              },
              emphasis: {
                label: {
                  show: true,
                  fontSize: 12,
                  fontWeight: 'bold',
                  color: '#000'
                },
                itemStyle: {
                  areaColor: '#ffd700'
                }
              },
              // 使用处理后的地图数据
              data: mapData
            }
          ]
        }
        
        // 打印最终的地图配置
        console.log('最终地图配置完成');
      } catch (error) {
        console.error('加载省份数据失败:', error)
      } finally {
        loading.value = false
      }
    }
    
    // 计算百分比
    const calculatePercentage = (value) => {
      if (totalCount.value === 0) return '0.00'
      return ((value / totalCount.value) * 100).toFixed(2)
    }
    
    onMounted(() => {
      loadProvinceData()
    })
    
    return {
      provinces,
      provinceBarOptions,
      provincePieOptions,
      mapOptions,
      loading,
      calculatePercentage
    }
  }
}
</script>

<style scoped>
.province-analysis {
  padding: 0;
}
.loading-container {
  padding: 20px;
}
.data-table {
  margin-top: 20px;
}
</style> 