# Spider和Spark说明文档

仅用于学习交流

## 1. 环境配置

### 使用Docker Compose启动环境

使用以下docker-compose配置启动Spark环境和MySQL数据库：

```yaml
services:
  spark-master:
    image: bitnami/spark:latest
    container_name: spark-master
    hostname: spark-master
    environment:
      - SPARK_MODE=master
      - SPARK_MASTER_HOST=spark-master
    ports:
      - "7077:7077"    # Spark Master 端口
      - "8080:8080"    # Web UI
    networks:
      - spark-network

  spark-worker:
    image: bitnami/spark:latest
    container_name: spark-worker
    hostname: spark-worker
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
    ports:
      - "8081:8081"    # Worker Web UI
    networks:
      - spark-network

  jupyter:
    image: jupyter/pyspark-notebook:latest
    container_name: jupyter-notebook
    depends_on:
      - spark-master
    ports:
      - "8888:8888"    # Jupyter Notebook 访问端口
    volumes:
      - ./notebooks:/home/jovyan/work  # 绑定本地 notebook 目录
    networks:
      - spark-network
  
  mysql:
    image: mysql:8
    container_name: mysql
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: testdb
      MYSQL_USER: user
      MYSQL_PASSWORD: password
    ports:
      - "3306:3306"
    networks:
      - spark-network
    volumes:
      - ./mysql-data:/var/lib/mysql

networks:
  spark-network:
    driver: bridge
```

## 2. 项目使用流程

### 2.1 Spider爬取考研网站数据

```bash
cd spider
python kaoyan_spider.py
```

### 2.2 Spark数据处理与分析

```bash
cd spark
# 查看详细说明
cat spark/README.md
```

### 2.3 Web应用运行

运行Flask项目及Vue前端，详细说明请参考：
```bash
cat README_Flask_VUE.md
```

## 3. Spider - 网络爬虫框架

### 3.1 简介
Spider是一种网络爬虫框架，用于从网站自动提取数据。它允许开发者以结构化的方式收集网页信息，无需手动浏览网站。

### 3.2 主要特点
- **自动化数据提取**：能够自动从网页提取所需信息
- **分布式爬取**：支持多线程、分布式系统进行大规模爬取
- **定制化**：可根据不同网站结构自定义爬取规则
- **数据处理**：支持数据清洗、转换和存储功能

### 3.3 常见爬虫框架
- **Scrapy**：Python的强大爬虫框架，功能全面
- **Beautiful Soup**：用于解析HTML和XML文档的Python库
- **Selenium**：自动化浏览器工具，适合爬取JavaScript渲染的页面
- **Puppeteer**：Node.js库，提供高级API控制Chrome/Chromium

### 3.4 应用场景
- 数据挖掘与分析
- 市场调研
- 价格监控
- 内容聚合
- 搜索引擎索引

## 4. Spark - 大数据处理框架

### 4.1 简介
Apache Spark是一个快速、通用的分布式计算系统，为大规模数据处理提供高效解决方案。它扩展了MapReduce模型，支持更多计算类型。

### 4.2 主要特点
- **速度**：比传统的Hadoop MapReduce快10-100倍
- **易用性**：支持Java、Scala、Python和R的API
- **通用性**：集成了SQL查询、流处理、机器学习和图计算
- **内存计算**：利用内存计算提高性能
- **容错性**：自动恢复失败的任务

### 4.3 核心组件
- **Spark Core**：基础引擎，负责内存管理、任务调度等
- **Spark SQL**：用于结构化数据处理
- **Spark Streaming**：实时数据流处理
- **MLlib**：机器学习库
- **GraphX**：图计算框架

### 4.4 应用场景
- 大规模数据处理
- 实时数据分析
- 机器学习模型训练
- 复杂的ETL流程
- 推荐系统
