import axios from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 响应拦截器
api.interceptors.response.use(
  response => {
    const res = response.data
    if (res.code !== 200) {
      console.error('API错误:', res.message || '未知错误')
      return Promise.reject(new Error(res.message || '未知错误'))
    } else {
      return res.data
    }
  },
  error => {
    console.error('网络错误:', error)
    return Promise.reject(error)
  }
)

// 封装API接口
export default {
  // 获取热门学校Top20
  getTopSchools() {
    return api.get('/top-schools')
  },
  
  // 获取热门专业Top20
  getTopMajors() {
    return api.get('/top-majors')
  },
  
  // 获取热门省份Top20
  getTopProvinces() {
    return api.get('/top-provinces')
  },
  
  // 获取所有统计数据
  getAllStats() {
    return api.get('/all-stats')
  }
} 