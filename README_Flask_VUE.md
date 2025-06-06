# 考研数据分析可视化系统

## 项目概述

本项目是一个基于Spark的考研数据分析系统，利用大数据技术分析考研相关数据，包括学校信息、专业信息、录取情况等，并通过Flask后端和Vue前端实现了数据可视化展示。系统主要展示考研热门学校TOP20、考研热门专业TOP20和考研热门省份TOP20的分析结果。

## 系统架构

系统采用前后端分离架构：

1. **后端**：
   - 使用Python + Flask构建API服务
   - 基于Spark进行大数据分析
   - 使用Pandas等库进行数据处理

2. **前端**：
   - 使用Vue.js框架
   - 使用Element Plus组件库
   - 使用ECharts实现数据可视化

## 主要功能

### 1. 数据总览

- 展示考研热门学校、专业和省份的统计信息
- 提供考研热门学校、专业Top10可视化图表
- 展示省份分布饼图

### 2. 学校分析

- 展示考研热门学校Top20柱状图
- 提供学校详细信息表格，包括申请人数、学校类型、985/211标识等
- 支持表格排序和筛选

### 3. 专业分析

- 展示考研热门专业Top20柱状图
- 提供专业类别占比饼图
- 展示专业详细信息表格，包括专业代码、名称、申请人数、专业类别等

### 4. 省份分析

- 展示考研热门省份Top20柱状图和饼图
- 提供中国地图可视化，直观展示各省份考研热度
- 显示省份详细信息表格，包括申请人数和占比

## 技术栈

### 后端技术

- **Flask**：Web框架
- **Flask-CORS**：处理跨域请求
- **PySpark**：大数据处理引擎
- **Pandas**：数据分析库
- **Matplotlib/Seaborn**：数据可视化库
- **Scikit-learn**：机器学习库

### 前端技术

- **Vue3**：前端框架
- **Vue Router**：路由管理
- **Element Plus**：UI组件库
- **ECharts**：图表可视化库
- **Axios**：HTTP客户端

## 目录结构

```
kaoyan-analysis/
├── app/                   # Flask应用目录
│   ├── api/               # API接口
│   │   ├── __init__.py    # API蓝图初始化
│   │   ├── data_service.py # 数据服务
│   │   └── routes.py      # API路由
│   ├── static/            # 静态资源
│   ├── templates/         # 模板文件
│   └── __init__.py        # Flask应用初始化
├── frontend/              # Vue.js前端目录
│   ├── src/               # 源代码
│   │   ├── api/           # API调用
│   │   ├── components/    # 组件
│   │   ├── views/         # 视图
│   │   └── router/        # 路由
│   └── public/            # 公共资源
└── run.py                 # Flask应用启动文件
```

## 接口说明

### 1. 获取热门学校Top20

- URL: `/api/top-schools`
- 方法: GET
- 响应: JSON格式的学校排名数据

### 2. 获取热门专业Top20

- URL: `/api/top-majors`
- 方法: GET
- 响应: JSON格式的专业排名数据

### 3. 获取热门省份Top20

- URL: `/api/top-provinces`
- 方法: GET
- 响应: JSON格式的省份排名数据

### 4. 获取所有统计数据

- URL: `/api/all-stats`
- 方法: GET
- 响应: JSON格式的所有统计数据

## 启动说明

### 后端启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Flask应用
python run.py
```

### 前端启动

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve

# 打包生产环境
npm run build
```

## 访问地址

- 前端访问地址：http://localhost:8080
- 后端API地址：http://localhost:5000/api
