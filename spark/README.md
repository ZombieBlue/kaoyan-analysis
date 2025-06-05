# 考研数据分析与预测

本目录包含使用Spark进行考研数据分析和预测的相关脚本。

## 文件说明

- `data_cleaner.py`: 数据清洗脚本，用于从CSV文件读取原始数据并进行清洗转换
- `data_analyzer.py`: 数据分析脚本，使用Spark进行多维度分析和机器学习模型训练
- `admission_predictor.py`: 录取概率预测工具，使用训练好的模型预测考生录取概率

## 使用步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据清洗

运行数据清洗脚本，将原始数据进行清洗和格式化：

```bash
python data_cleaner.py
```

清洗后的数据将保存在`data/clean`目录下。

### 3. 数据分析与模型训练

运行数据分析脚本，进行多维度分析并训练机器学习模型：

```bash
python data_analyzer.py
```

分析结果将保存在`data/analysis`目录下，模型将保存在`models`目录下。

### 4. 录取预测

#### 学校和专业查询

查询学校信息：

```bash
python admission_predictor.py search_school 北京
```

查询专业信息：

```bash
python admission_predictor.py search_major 计算机
```

#### 录取概率预测

预测考生被某学校某专业录取的概率：

```bash
python admission_predictor.py predict_prob 10001 0812 360 --year 2025
```

参数说明：
- `10001`：学校ID
- `0812`：专业代码
- `360`：考生总分
- `--year 2025`：预测年份（可选，默认2025）

#### 分数线预测

预测某学校某专业的录取分数线：

```bash
python admission_predictor.py predict_score 10001 0812 --year 2025 --target min_score
```

参数说明：
- `10001`：学校ID
- `0812`：专业代码
- `--year 2023`：预测年份（可选，默认2023）
- `--target min_score`：预测目标，可选值为`min_score`（最低分）或`avg_score`（平均分），默认为`min_score`

## 功能特点

### 数据分析功能

- **专业热度分析**：分析各专业报考人数、热门专业排名和专业类别分布
- **录取趋势分析**：分析近年来考研录取分数变化趋势，985/211与普通院校对比
- **分数分布分析**：分析各省份、各学校类型的分数分布差异

### 机器学习预测功能

- **分数线预测**：预测学校专业的最低录取分数线和平均录取分数线
- **录取概率预测**：根据考生分数预测被录取的概率

## 模型性能

系统训练了多种模型，包括线性回归（Linear Regression）、随机森林（Random Forest）和梯度提升树（Gradient Boosted Trees），并根据评估指标选择最佳模型。

- **分数线预测模型**：RMSE（均方根误差）< 10分
- **录取概率预测模型**：准确率（Accuracy）> 0.8 