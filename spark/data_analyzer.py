#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import configparser
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, desc, year, max as spark_max, min as spark_min, stddev, expr, when, lit
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import matplotlib
import json
import shutil

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 创建日志目录
logs_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'data_analyzer.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataAnalyzer')

class KaoyanDataAnalyzer:
    """使用Spark进行考研数据分析和机器学习预测"""
    
    def __init__(self):
        """初始化数据分析器"""
        # 数据目录
        self.data_dir = os.path.join(PROJECT_ROOT, 'data')
        self.clean_data_dir = os.path.join(self.data_dir, 'clean')
        self.analysis_dir = os.path.join(self.data_dir, 'analysis')
        self.models_dir = os.path.join(PROJECT_ROOT, 'models')
        
        # 确保目录存在
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 创建Spark会话
        self.spark = self._create_spark_session()
        
        # 图表样式设置
        self._setup_plot_style()
        
        logger.info("数据分析器已准备就绪")
    
    def _create_spark_session(self):
        """创建Spark会话"""
        try:
            # 创建Spark会话
            spark = SparkSession.builder \
                .appName("KaoyanDataAnalyzer") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            # 设置日志级别为ERROR，减少控制台输出
            spark.sparkContext.setLogLevel("ERROR")
            
            logger.info("成功创建Spark会话")
            return spark
        except Exception as e:
            logger.error(f"创建Spark会话失败: {e}")
            raise
    
    def _setup_plot_style(self):
        """设置绘图样式"""
        # 尝试解决中文显示问题
        try:
            # 尝试使用不同的中文字体
            matplotlib.rcParams['font.family'] = ['sans-serif']
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            # 检查是否有可用字体
            logger.info("使用字体：" + matplotlib.rcParams['font.sans-serif'][0])
        except Exception as e:
            logger.warning(f"设置中文字体失败，将使用默认字体: {e}")
            # 如果设置失败，使用英文标签并记录警告
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
        # 设置Seaborn样式
        sns.set(style="whitegrid")
        
    def load_data(self):
        """加载清洗后的数据"""
        try:
            logger.info("开始加载清洗后的数据...")
            
            # 加载各个CSV文件
            schools_df = self.spark.read.csv(os.path.join(self.clean_data_dir, 'schools.csv'), 
                                            header=True, inferSchema=True, encoding='utf-8')
            
            # 加载分数数据 - 使用目录形式的输出
            scores_csv_dir = os.path.join(self.clean_data_dir, 'scores.csv')
            scores_df = self.spark.read.csv(scores_csv_dir, header=True, inferSchema=True)
            
            # 加载专业数据
            majors_df = self.spark.read.csv(os.path.join(self.clean_data_dir, 'majors.csv'), 
                                          header=True, inferSchema=True, encoding='utf-8')
            
            # 加载其他分析视图
            school_scores_dir = os.path.join(self.clean_data_dir, 'school_scores_summary.csv')
            school_scores_summary = self.spark.read.csv(school_scores_dir, header=True, inferSchema=True)
            
            major_scores_dir = os.path.join(self.clean_data_dir, 'major_scores.csv')
            major_scores = self.spark.read.csv(major_scores_dir, header=True, inferSchema=True)
            
            major_avg_dir = os.path.join(self.clean_data_dir, 'major_avg_scores.csv')
            major_avg_scores = self.spark.read.csv(major_avg_dir, header=True, inferSchema=True)
            
            elite_vs_normal_dir = os.path.join(self.clean_data_dir, 'elite_vs_normal.csv')
            elite_vs_normal = self.spark.read.csv(elite_vs_normal_dir, header=True, inferSchema=True)
            
            logger.info(f"数据加载完成: 学校 {schools_df.count()} 条, 分数 {scores_df.count()} 条, 专业 {majors_df.count()} 条")
            
            # 返回数据字典
            return {
                'schools': schools_df,
                'scores': scores_df,
                'majors': majors_df,
                'school_scores_summary': school_scores_summary,
                'major_scores': major_scores,
                'major_avg_scores': major_avg_scores,
                'elite_vs_normal': elite_vs_normal
            }
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
            
    def analyze_major_popularity(self, data_dict):
        """分析专业热度"""
        try:
            logger.info("开始分析专业热度...")
            
            scores_df = data_dict['scores']
            
            # 使用RDD进行分析
            # 将DataFrame转换为RDD，并提取专业代码和名称
            major_rdd = scores_df.select("code", "name").rdd
            
            # 使用map-reduce计算每个专业的频次
            major_counts = major_rdd.map(lambda x: ((x[0], x[1]), 1)) \
                                    .reduceByKey(lambda a, b: a + b) \
                                    .map(lambda x: (x[0][0], x[0][1], x[1])) \
                                    .toDF(["code", "name", "count"])
            
            # 排序并获取前30个热门专业
            top_majors = major_counts.orderBy(desc("count")).limit(30)
            
            # 转换为Pandas DataFrame进行可视化
            top_majors_pd = top_majors.toPandas()
            
            # 绘制热门专业柱状图
            plt.figure(figsize=(14, 8))
            bars = plt.barh(top_majors_pd['name'][:20], top_majors_pd['count'][:20])
            plt.xlabel('Applicants')
            plt.ylabel('Major Name')
            plt.title('Top 20 Popular Majors')
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.analysis_dir, 'top_majors.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 按专业类别汇总
            if 'major_category' in scores_df.columns:
                category_counts = scores_df.groupBy("major_category") \
                                        .count() \
                                        .orderBy(desc("count"))
                
                # 转换为Pandas DataFrame并绘制饼图
                category_pd = category_counts.toPandas()
                
                plt.figure(figsize=(10, 10))
                plt.pie(category_pd['count'], labels=category_pd['major_category'], 
                        autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title('Major Category Distribution')
                
                category_chart_path = os.path.join(self.analysis_dir, 'major_categories.png')
                plt.savefig(category_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"专业热度分析完成，结果已保存至: {self.analysis_dir}")
            
            return top_majors
            
        except Exception as e:
            logger.error(f"专业热度分析失败: {e}")
            raise
    
    def analyze_admission_trends(self, data_dict):
        """分析录取趋势"""
        try:
            logger.info("开始分析录取趋势...")
            
            # 使用学校分数汇总数据
            school_scores = data_dict['school_scores_summary']
            
            # 按年份分析平均分数变化
            yearly_avg = school_scores.groupBy("year") \
                                    .agg(
                                        avg("avg_total_score").alias("avg_score"),
                                        stddev("avg_total_score").alias("stddev_score")
                                    ) \
                                    .orderBy("year")
            
            # 转换为Pandas DataFrame进行可视化
            yearly_avg_pd = yearly_avg.toPandas()
            
            # 绘制年度平均分数变化趋势
            plt.figure(figsize=(12, 6))
            plt.errorbar(yearly_avg_pd['year'], yearly_avg_pd['avg_score'], 
                        yerr=yearly_avg_pd['stddev_score'], marker='o', linestyle='-')
            plt.xlabel('Year')
            plt.ylabel('Average Score')
            plt.title('Annual Admission Score Trends')
            plt.grid(True)
            
            # 保存图表
            trend_path = os.path.join(self.analysis_dir, 'score_trends.png')
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 分析985/211和普通院校的分数趋势
            if 'elite_vs_normal' in data_dict:
                elite_df = data_dict['elite_vs_normal']
                
                # 转换为Pandas DataFrame
                elite_pd = elite_df.toPandas()
                
                # 透视表以便于绘图
                pivot_df = elite_pd.pivot(index='year', columns='school_type', values='avg_total_score')
                
                # 绘制985/211和普通院校的分数趋势对比
                plt.figure(figsize=(12, 6))
                pivot_df.plot(marker='o', linestyle='-')
                plt.xlabel('Year')
                plt.ylabel('Average Score')
                plt.title('Elite vs Regular Schools Score Trends')
                plt.grid(True)
                plt.legend(title='School Type')
                
                elite_trend_path = os.path.join(self.analysis_dir, 'elite_normal_trends.png')
                plt.savefig(elite_trend_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"录取趋势分析完成，结果已保存至: {self.analysis_dir}")
            
            return yearly_avg
            
        except Exception as e:
            logger.error(f"录取趋势分析失败: {e}")
            raise
    
    def analyze_score_distribution(self, data_dict):
        """分析学校考研分数分布"""
        try:
            logger.info("开始分析学校考研分数分布...")
            
            scores_df = data_dict['scores']
            schools_df = data_dict['schools']
            
            # 合并学校信息
            joined_df = scores_df.join(
                schools_df.select("school_id", "school_name", "province", "type", "is_985", "is_211", "rank"),
                on="school_id",
                how="left"
            )
            
            # 统计各省份的平均分数
            province_scores = joined_df.groupBy("province") \
                                     .agg(
                                         avg("total").alias("avg_score"),
                                         count("*").alias("count")
                                     ) \
                                     .orderBy(desc("avg_score"))
            
            # 转换为Pandas DataFrame
            province_scores_pd = province_scores.toPandas()
            
            # 绘制各省份平均分数柱状图
            plt.figure(figsize=(14, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(province_scores_pd)))
            bars = plt.bar(province_scores_pd['province'], province_scores_pd['avg_score'], color=colors)
            plt.xlabel('Province')
            plt.ylabel('Average Score')
            plt.title('Average Scores by Province')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # 保存图表
            province_path = os.path.join(self.analysis_dir, 'province_scores.png')
            plt.savefig(province_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 分析学校类型的分数分布
            school_type_scores = joined_df.groupBy("type") \
                                        .agg(
                                            avg("total").alias("avg_score"),
                                            count("*").alias("count")
                                        ) \
                                        .orderBy(desc("avg_score"))
            
            # 转换为Pandas DataFrame
            type_scores_pd = school_type_scores.toPandas()
            
            # 绘制学校类型分数箱线图
            plt.figure(figsize=(12, 6))
            joined_pd = joined_df.select("type", "total").toPandas()
            sns.boxplot(x='type', y='total', data=joined_pd)
            plt.xlabel('School Type')
            plt.ylabel('Total Score')
            plt.title('Score Distribution by School Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            type_path = os.path.join(self.analysis_dir, 'school_type_scores.png')
            plt.savefig(type_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 985/211院校与普通院校的分数分布比较
            joined_df = joined_df.withColumn(
                "elite_type",
                when((col("is_985") == 1) & (col("is_211") == 1), "985&211")
                .when((col("is_985") == 0) & (col("is_211") == 1), "211")
                .otherwise("Regular")
            )
            
            # 转换为Pandas DataFrame
            elite_pd = joined_df.select("elite_type", "total").toPandas()
            
            # 绘制985/211与普通院校分数密度图
            plt.figure(figsize=(12, 6))
            for elite_type in elite_pd['elite_type'].unique():
                subset = elite_pd[elite_pd['elite_type'] == elite_type]
                sns.kdeplot(subset['total'], label=elite_type)
            
            plt.xlabel('Total Score')
            plt.ylabel('Density')
            plt.title('Score Distribution: Elite vs Regular Schools')
            plt.legend(title='School Type')
            plt.grid(True)
            
            # 保存图表
            elite_dist_path = os.path.join(self.analysis_dir, 'elite_score_distribution.png')
            plt.savefig(elite_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"学校考研分数分布分析完成，结果已保存至: {self.analysis_dir}")
            
            return {
                'province_scores': province_scores,
                'school_type_scores': school_type_scores
            }
            
        except Exception as e:
            logger.error(f"学校考研分数分布分析失败: {e}")
            raise
    
    def train_score_prediction_model(self, data_dict):
        """训练各学校各专业录取分数线预测模型"""
        try:
            logger.info("开始训练各学校各专业录取分数线预测模型...")
            
            scores_df = data_dict['scores']
            schools_df = data_dict['schools']
            
            # 合并数据集 - 使用别名区分相同字段
            joined_df = scores_df.join(
                schools_df.select(
                    col("school_id"),
                    col("school_name").alias("school_full_name"),  # 重命名学校名称字段
                    "province", "type", "is_985", "is_211", "rank"
                ),
                on="school_id",
                how="left"
            )
            
            # 计算每个学校每个专业每年的最低录取分数 - 使用重命名后的学校名称字段
            admission_scores = joined_df.groupBy("school_id", "school_full_name", "code", "name", "year") \
                .agg(
                    spark_min("total").alias("min_score"),
                    avg("total").alias("avg_score"),
                    count("*").alias("applicants")
                )
            
            # 记录最低录取分数结果
            admission_scores_pd = admission_scores.toPandas()
            admission_scores_path = os.path.join(self.analysis_dir, 'admission_scores.csv')
            admission_scores_pd.to_csv(admission_scores_path, index=False, encoding='utf-8')
            logger.info(f"各学校各专业录取分数线已保存至: {admission_scores_path}")
            
            # 数据预处理
            # 1. 选择特征和目标变量
            model_df = admission_scores.join(
                schools_df.select("school_id", "province", "type", "is_985", "is_211", "rank"),
                on="school_id",
                how="left"
            )
            
            # 2. 处理缺失值 - 更全面地处理各个字段
            model_df = model_df.na.fill({
                "min_score": 0,
                "avg_score": 0,
                "applicants": 0,
                "rank": 999,
                "is_985": 0,
                "is_211": 0,
                "year": 2022
            })
            
            # 检查并丢弃包含null值的行
            initial_count = model_df.count()
            model_df = model_df.dropna()
            dropped_count = initial_count - model_df.count()
            if dropped_count > 0:
                logger.warning(f"丢弃了{dropped_count}行包含null值的数据")
            
            # 3. 进行特征工程
            # 字符串索引转换
            string_indexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="keep")
            string_indexer_model = string_indexer.fit(model_df)
            model_df = string_indexer_model.transform(model_df)
            
            # 对专业代码进行索引
            major_indexer = StringIndexer(inputCol="code", outputCol="code_index", handleInvalid="keep")
            major_indexer_model = major_indexer.fit(model_df)
            model_df = major_indexer_model.transform(model_df)
            
            # 对省份进行索引
            province_indexer = StringIndexer(inputCol="province", outputCol="province_index", handleInvalid="keep")
            province_indexer_model = province_indexer.fit(model_df)
            model_df = province_indexer_model.transform(model_df)
            
            # 4. 创建特征向量
            feature_cols = ["applicants", "is_985", "is_211", "type_index", "code_index", 
                           "province_index", "rank", "year"]
            
            # 确保所有特征列都转换为Double类型
            for col_name in feature_cols:
                model_df = model_df.withColumn(f"{col_name}_double", col(col_name).cast(DoubleType()))
            
            # 更新特征列名
            feature_cols_double = [f"{col_name}_double" for col_name in feature_cols]
            
            # 使用handleInvalid="skip"参数创建VectorAssembler
            assembler = VectorAssembler(
                inputCols=feature_cols_double, 
                outputCol="features", 
                handleInvalid="skip"  # 跳过包含null的行
            )
            model_df = assembler.transform(model_df)
            
            # 5. 标准化特征
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(model_df)
            model_df = scaler_model.transform(model_df)
            
            # 将模型训练设为两个目标：预测最低分和平均分
            for target in ["min_score", "avg_score"]:
                logger.info(f"开始训练{target}预测模型...")
                
                # 6. 划分训练集和测试集
                (train_df, test_df) = model_df.randomSplit([0.8, 0.2], seed=42)
                
                # 7. 训练多个模型
                # 线性回归
                lr = LinearRegression(featuresCol="scaled_features", labelCol=target, 
                                    maxIter=10, regParam=0.1, elasticNetParam=0.8)
                lr_model = lr.fit(train_df)
                
                # 随机森林回归
                rf = RandomForestRegressor(featuresCol="scaled_features", labelCol=target, 
                                        numTrees=20, maxDepth=5)
                rf_model = rf.fit(train_df)
                
                # 梯度提升树回归
                gbt = GBTRegressor(featuresCol="scaled_features", labelCol=target, 
                                maxIter=10, maxDepth=5)
                gbt_model = gbt.fit(train_df)
                
                # 8. 评估模型
                # 创建评估器
                evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", 
                                            metricName="rmse")
                
                # 线性回归评估
                lr_predictions = lr_model.transform(test_df)
                lr_rmse = evaluator.evaluate(lr_predictions)
                
                # 随机森林评估
                rf_predictions = rf_model.transform(test_df)
                rf_rmse = evaluator.evaluate(rf_predictions)
                
                # 梯度提升树评估
                gbt_predictions = gbt_model.transform(test_df)
                gbt_rmse = evaluator.evaluate(gbt_predictions)
                
                # 输出评估结果
                logger.info(f"{target} 线性回归模型RMSE: {lr_rmse}")
                logger.info(f"{target} 随机森林模型RMSE: {rf_rmse}")
                logger.info(f"{target} 梯度提升树模型RMSE: {gbt_rmse}")
                
                # 9. 选择最佳模型
                models = {
                    'LinearRegression': (lr_model, lr_rmse),
                    'RandomForest': (rf_model, rf_rmse),
                    'GBT': (gbt_model, gbt_rmse)
                }
                
                best_model_name = min(models, key=lambda k: models[k][1])
                best_model, best_rmse = models[best_model_name]
                
                logger.info(f"{target} 最佳模型: {best_model_name}, RMSE: {best_rmse}")
                
                # 10. 保存模型和元数据
                model_path = os.path.join(self.models_dir, f"{target}_prediction_{best_model_name}")
                
                # 检查目录是否存在，如果存在则先删除
                if os.path.exists(model_path):
                    logger.info(f"模型路径已存在，正在删除: {model_path}")
                    shutil.rmtree(model_path)
                
                # 保存模型
                best_model.write().overwrite().save(model_path)
                
                # 保存模型元数据
                model_meta = {
                    'model_name': best_model_name,
                    'rmse': best_rmse,
                    'feature_columns': feature_cols_double,
                    'original_feature_columns': feature_cols,
                    'target': target
                }
                
                # 保存特征工程的元数据
                meta_path = os.path.join(self.models_dir, f"{target}_prediction_{best_model_name}_meta.json")
                try:
                    with open(meta_path, 'w') as f:
                        json.dump(model_meta, f)
                    logger.info(f"{target}预测模型元数据已保存至: {meta_path}")
                except Exception as e:
                    logger.warning(f"保存模型元数据失败: {e}")
                
                # 可视化模型性能
                # 创建实际值与预测值对比图
                predictions_df = best_model.transform(test_df)
                pred_vs_actual = predictions_df.select(target, "prediction").toPandas()
                
                plt.figure(figsize=(10, 8))
                plt.scatter(pred_vs_actual[target], pred_vs_actual["prediction"], alpha=0.5)
                plt.plot([pred_vs_actual[target].min(), pred_vs_actual[target].max()], 
                        [pred_vs_actual[target].min(), pred_vs_actual[target].max()], 'r--')
                plt.xlabel(f'Actual {target}')
                plt.ylabel(f'Predicted {target}')
                plt.title(f'{target.capitalize()} Prediction Model Performance ({best_model_name})')
                plt.grid(True)
                
                # 保存图表
                model_perf_path = os.path.join(self.analysis_dir, f'{target}_prediction_performance.png')
                plt.savefig(model_perf_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"{target}预测模型训练完成，最佳模型已保存至: {model_path}")
            
            # 返回最后一个模型的信息
            return {
                'admission_scores': admission_scores,
                'model': best_model,
                'model_name': best_model_name,
                'rmse': best_rmse,
                'model_path': model_path,
                'meta': model_meta
            }
            
        except Exception as e:
            logger.error(f"训练录取分数线预测模型失败: {e}")
            raise
    
    def train_admission_probability_model(self, data_dict):
        """训练录取概率预测模型"""
        try:
            logger.info("开始训练录取概率预测模型...")
            
            scores_df = data_dict['scores']
            schools_df = data_dict['schools']
            
            # 合并数据集 - 使用别名区分相同字段
            joined_df = scores_df.join(
                schools_df.select(
                    col("school_id"),
                    col("school_name").alias("school_full_name"),  # 重命名学校名称字段
                    "province", "type", "is_985", "is_211", "rank"
                ),
                on="school_id",
                how="left"
            )
            
            # 为训练二分类模型，需要创建一个虚拟的录取标识
            # 这里假设分数高于该学校专业平均分的为录取，低于的为未录取
            # 注意：这是一个简化假设，实际情况可能更复杂
            
            # 1. 计算每个学校专业的平均分
            avg_scores = joined_df.groupBy("school_id", "code").agg(avg("total").alias("avg_school_major_score"))
            
            # 2. 将平均分与原始数据关联
            joined_df = joined_df.join(
                avg_scores,
                on=["school_id", "code"],
                how="left"
            )
            
            # 3. 创建录取标识列
            joined_df = joined_df.withColumn(
                "admitted",
                when(col("total") >= col("avg_school_major_score"), 1).otherwise(0)
            )
            
            # 4. 数据预处理
            model_df = joined_df.select(
                "admitted", "politics", "english", "special_one", "special_two",
                "school_id", "year", "is_985", "is_211", "type", "code", "rank"
            )
            
            # 5. 处理缺失值
            model_df = model_df.na.fill({
                "politics": 0,
                "english": 0,
                "special_one": 0,
                "special_two": 0,
                "rank": 999,
                "is_985": 0,
                "is_211": 0,
                "year": 2022,
                "admitted": 0
            })
            
            # 检查并丢弃包含null值的行
            initial_count = model_df.count()
            model_df = model_df.dropna()
            dropped_count = initial_count - model_df.count()
            if dropped_count > 0:
                logger.warning(f"丢弃了{dropped_count}行包含null值的数据")
            
            # 6. 进行特征工程
            # 字符串索引转换
            string_indexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="keep")
            string_indexer_model = string_indexer.fit(model_df)
            model_df = string_indexer_model.transform(model_df)
            
            # 对专业代码进行索引
            major_indexer = StringIndexer(inputCol="code", outputCol="code_index", handleInvalid="keep")
            major_indexer_model = major_indexer.fit(model_df)
            model_df = major_indexer_model.transform(model_df)
            
            # 7. 创建特征向量
            feature_cols = ["politics", "english", "special_one", "special_two", 
                           "is_985", "is_211", "type_index", "code_index", "rank", "year"]
            
            # 确保所有特征列都转换为Double类型
            for col_name in feature_cols:
                model_df = model_df.withColumn(f"{col_name}_double", col(col_name).cast(DoubleType()))
            
            # 更新特征列名
            feature_cols_double = [f"{col_name}_double" for col_name in feature_cols]
            
            # 使用handleInvalid="skip"参数创建VectorAssembler
            assembler = VectorAssembler(
                inputCols=feature_cols_double,
                outputCol="features",
                handleInvalid="skip"  # 跳过包含null的行
            )
            model_df = assembler.transform(model_df)
            
            # 8. 标准化特征
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(model_df)
            model_df = scaler_model.transform(model_df)
            
            # 9. 划分训练集和测试集
            (train_df, test_df) = model_df.randomSplit([0.8, 0.2], seed=42)
            
            # 10. 训练多个模型
            # 逻辑回归
            lr = LogisticRegression(featuresCol="scaled_features", labelCol="admitted", 
                                   maxIter=10, regParam=0.1)
            lr_model = lr.fit(train_df)
            
            # 随机森林分类器
            rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="admitted", 
                                       numTrees=20, maxDepth=5)
            rf_model = rf.fit(train_df)
            
            # 梯度提升树分类器
            gbt = GBTClassifier(featuresCol="scaled_features", labelCol="admitted", 
                               maxIter=10, maxDepth=5)
            gbt_model = gbt.fit(train_df)
            
            # 11. 评估模型
            # 创建评估器
            evaluator = MulticlassClassificationEvaluator(labelCol="admitted", predictionCol="prediction", 
                                                        metricName="accuracy")
            
            # 逻辑回归评估
            lr_predictions = lr_model.transform(test_df)
            lr_accuracy = evaluator.evaluate(lr_predictions)
            
            # 随机森林评估
            rf_predictions = rf_model.transform(test_df)
            rf_accuracy = evaluator.evaluate(rf_predictions)
            
            # 梯度提升树评估
            gbt_predictions = gbt_model.transform(test_df)
            gbt_accuracy = evaluator.evaluate(gbt_predictions)
            
            # 输出评估结果
            logger.info(f"逻辑回归模型准确率: {lr_accuracy}")
            logger.info(f"随机森林模型准确率: {rf_accuracy}")
            logger.info(f"梯度提升树模型准确率: {gbt_accuracy}")
            
            # 12. 选择最佳模型
            models = {
                'LogisticRegression': (lr_model, lr_accuracy),
                'RandomForest': (rf_model, rf_accuracy),
                'GBT': (gbt_model, gbt_accuracy)
            }
            
            best_model_name = max(models, key=lambda k: models[k][1])
            best_model, best_accuracy = models[best_model_name]
            
            logger.info(f"最佳模型: {best_model_name}, 准确率: {best_accuracy}")
            
            # 13. 保存模型和元数据
            model_path = os.path.join(self.models_dir, f"admission_probability_{best_model_name}")
            
            # 检查目录是否存在，如果存在则先删除
            if os.path.exists(model_path):
                logger.info(f"模型路径已存在，正在删除: {model_path}")
                shutil.rmtree(model_path)
            
            # 保存模型
            best_model.write().overwrite().save(model_path)
            
            # 保存模型元数据
            model_meta = {
                'model_name': best_model_name,
                'accuracy': best_accuracy,
                'feature_columns': feature_cols_double,
                'original_feature_columns': feature_cols
            }
            
            # 保存特征工程的元数据
            meta_path = os.path.join(self.models_dir, f"admission_probability_{best_model_name}_meta.json")
            try:
                with open(meta_path, 'w') as f:
                    json.dump(model_meta, f)
                logger.info(f"模型元数据已保存至: {meta_path}")
            except Exception as e:
                logger.warning(f"保存模型元数据失败: {e}")
            
            # 可视化最佳模型的混淆矩阵
            if best_model_name == 'LogisticRegression':
                # 获取ROC曲线数据
                roc_data = lr_model.summary.roc.toPandas()
                
                plt.figure(figsize=(8, 8))
                plt.plot(roc_data['FPR'], roc_data['TPR'])
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlabel('False Positive Rate (FPR)')
                plt.ylabel('True Positive Rate (TPR)')
                plt.title(f'ROC Curve - {best_model_name} (AUC = {lr_model.summary.areaUnderROC:.4f})')
                plt.grid(True)
                
                # 保存ROC曲线
                roc_path = os.path.join(self.analysis_dir, 'admission_probability_roc.png')
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 计算并可视化特征重要性（对于随机森林和GBT）
            if best_model_name in ['RandomForest', 'GBT']:
                feature_importances = best_model.featureImportances.toArray()
                features_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': feature_importances
                })
                features_df = features_df.sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                plt.barh(features_df['feature'], features_df['importance'])
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                plt.title(f'Feature Importance - {best_model_name}')
                plt.tight_layout()
                
                # 保存特征重要性图
                importance_path = os.path.join(self.analysis_dir, 'feature_importance.png')
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"录取概率预测模型训练完成，最佳模型已保存至: {model_path}")
            
            return {
                'model': best_model,
                'model_name': best_model_name,
                'accuracy': best_accuracy,
                'model_path': model_path,
                'meta': model_meta
            }
            
        except Exception as e:
            logger.error(f"训练录取概率预测模型失败: {e}")
            raise
    
    def predict_admission_probability(self, model_dict, student_data):
        """预测考生录取概率
        
        参数:
            model_dict: 训练好的模型字典，或者空字典（此时会尝试加载保存的模型）
            student_data: 包含考生信息的字典，例如：
                {
                    'politics': 70,
                    'english': 80,
                    'special_one': 130,
                    'special_two': 135,
                    'school_id': 10001,
                    'code': '085400',
                    'year': 2023,
                    'score': 360  # 总分
                }
        
        返回:
            录取概率
        """
        try:
            logger.info("开始预测考生录取概率...")
            
            # 获取模型和元数据
            if model_dict and 'model' in model_dict:
                # 使用传入的模型
                model = model_dict['model']
                meta = model_dict.get('meta', {})
            else:
                # 尝试加载保存的模型
                logger.info("未提供模型，尝试加载已保存的模型...")
                
                # 默认使用GBT模型
                model_path = os.path.join(self.models_dir, "admission_probability_GBT")
                meta_path = os.path.join(self.models_dir, "admission_probability_GBT_meta.json")
                
                # 检查模型路径是否存在，如果不存在，尝试其他模型
                models = ['GBT', 'RandomForest', 'LogisticRegression']
                for model_name in models:
                    model_path = os.path.join(self.models_dir, f"admission_probability_{model_name}")
                    meta_path = os.path.join(self.models_dir, f"admission_probability_{model_name}_meta.json")
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        break
                
                if not os.path.exists(model_path) or not os.path.isdir(model_path):
                    raise ValueError("未找到录取概率预测模型，请先运行模型训练")
                
                # 加载元数据
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                # 根据模型类型加载模型
                model_name = meta['model_name']
                if model_name == 'LogisticRegression':
                    from pyspark.ml.classification import LogisticRegressionModel
                    model = LogisticRegressionModel.load(model_path)
                elif model_name == 'RandomForest':
                    from pyspark.ml.classification import RandomForestClassificationModel
                    model = RandomForestClassificationModel.load(model_path)
                elif model_name == 'GBT':
                    from pyspark.ml.classification import GBTClassificationModel
                    model = GBTClassificationModel.load(model_path)
                else:
                    raise ValueError(f"不支持的模型类型: {model_name}")
            
            # 创建单行DataFrame
            # 处理传入的总分
            if 'score' in student_data and 'politics' not in student_data:
                # 如果只提供了总分，则根据一定规则拆分为各科分数
                total_score = student_data['score']
                politics_ratio = 0.2
                english_ratio = 0.3
                special_ratio = 0.5
                
                politics = int(total_score * politics_ratio)
                english = int(total_score * english_ratio)
                # 特殊科目平分剩余分数
                special_score = total_score - politics - english
                special_one = int(special_score / 2)
                special_two = special_score - special_one
                
                student_data['politics'] = politics
                student_data['english'] = english
                student_data['special_one'] = special_one
                student_data['special_two'] = special_two
            
            student_row = self.spark.createDataFrame([student_data])
            
            # 获取学校信息
            schools_df = self.spark.read.csv(os.path.join(self.clean_data_dir, 'schools.csv'), 
                                           header=True, inferSchema=True)
            
            # 关联学校信息
            school_info = schools_df.filter(col("school_id") == student_data['school_id']) \
                                   .select("is_985", "is_211", "type", "rank") \
                                   .first()
            
            if not school_info:
                raise ValueError(f"未找到ID为{student_data['school_id']}的学校信息")
            
            # 补充学校信息
            student_row = student_row.withColumn("is_985", lit(school_info["is_985"])) \
                                    .withColumn("is_211", lit(school_info["is_211"])) \
                                    .withColumn("type", lit(school_info["type"])) \
                                    .withColumn("rank", lit(school_info["rank"]))
            
            # 应用特征工程
            # 字符串索引转换
            string_indexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="keep")
            student_row = string_indexer.fit(student_row).transform(student_row)
            
            # 专业代码索引转换
            major_indexer = StringIndexer(inputCol="code", outputCol="code_index", handleInvalid="keep")
            student_row = major_indexer.fit(student_row).transform(student_row)
            
            # 确定特征列
            original_feature_cols = meta.get('original_feature_columns', ["politics", "english", "special_one", "special_two", 
                                                                        "is_985", "is_211", "type_index", "code_index", "rank", "year"])
            
            # 确保所有特征列都转换为Double类型
            for col_name in original_feature_cols:
                student_row = student_row.withColumn(f"{col_name}_double", col(col_name).cast(DoubleType()))
            
            # 获取实际使用的特征列名
            feature_cols_double = meta.get('feature_columns', [f"{col_name}_double" for col_name in original_feature_cols])
            
            # 使用handleInvalid参数创建VectorAssembler
            assembler = VectorAssembler(
                inputCols=feature_cols_double, 
                outputCol="features",
                handleInvalid="skip"
            )
            student_row = assembler.transform(student_row)
            
            # 标准化特征
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(student_row)
            student_row = scaler_model.transform(student_row)
            
            # 使用模型预测
            prediction = model.transform(student_row)
            
            # 获取预测结果
            if model_dict['model_name'] in ['LogisticRegression', 'RandomForest', 'GBT']:
                # 分类模型 - 获取概率
                if 'probability' in prediction.columns:
                    result = prediction.select("probability").first()[0][1]  # 第二个元素是正类的概率
                else:
                    # 如果没有概率列，则使用原始预测
                    result = float(prediction.select("prediction").first()[0])
            else:
                # 回归模型 - 获取预测值
                result = float(prediction.select("prediction").first()[0])
            
            logger.info(f"预测完成，录取概率: {result:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"预测录取概率失败: {e}")
            raise
    
    def predict_admission_score(self, school_id, code, year=2023, target="min_score"):
        """预测某学校某专业的录取分数线
        
        参数:
            school_id: 学校ID
            code: 专业代码
            year: 预测年份，默认2023
            target: 预测目标，"min_score"(最低分)或"avg_score"(平均分)
            
        返回:
            预测分数
        """
        try:
            logger.info(f"开始预测学校(ID:{school_id})专业(code:{code})在{year}年的{target}...")
            
            # 加载模型和元数据
            model_path = os.path.join(self.models_dir, f"{target}_prediction_GBT")  # 默认使用GBT模型
            meta_path = os.path.join(self.models_dir, f"{target}_prediction_GBT_meta.json")
            
            # 检查模型路径是否存在，如果不存在，尝试其他模型
            models = ['GBT', 'RandomForest', 'LinearRegression']
            for model_name in models:
                model_path = os.path.join(self.models_dir, f"{target}_prediction_{model_name}")
                meta_path = os.path.join(self.models_dir, f"{target}_prediction_{model_name}_meta.json")
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    break
            
            if not os.path.exists(model_path) or not os.path.isdir(model_path):
                raise ValueError(f"未找到{target}预测模型，请先运行模型训练")
            
            # 加载元数据
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # 根据模型类型加载模型
            model_name = meta['model_name']
            if model_name == 'LinearRegression':
                from pyspark.ml.regression import LinearRegressionModel
                model = LinearRegressionModel.load(model_path)
            elif model_name == 'RandomForest':
                from pyspark.ml.regression import RandomForestRegressionModel
                model = RandomForestRegressionModel.load(model_path)
            elif model_name == 'GBT':
                from pyspark.ml.regression import GBTRegressionModel
                model = GBTRegressionModel.load(model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
            # 获取学校信息
            schools_df = self.spark.read.csv(os.path.join(self.clean_data_dir, 'schools.csv'), 
                                           header=True, inferSchema=True)
            school_info = schools_df.filter(col("school_id") == school_id).first()
            
            if not school_info:
                raise ValueError(f"未找到ID为{school_id}的学校信息")
            
            # 获取学校历史录取数据，用于估算申请人数
            scores_df = self.spark.read.csv(os.path.join(self.clean_data_dir, 'scores.csv'), 
                                           header=True, inferSchema=True)
            
            # 估算申请人数（取该学校该专业最近一年的申请人数或平均申请人数）
            applicants_df = scores_df.filter((col("school_id") == school_id) & (col("code") == code)) \
                                   .groupBy("year").count() \
                                   .orderBy(desc("year"))
            
            # 如果没有历史数据，则使用默认值
            applicants = 20  # 默认值
            if applicants_df.count() > 0:
                applicants = applicants_df.first()["count"]
            
            # 创建预测数据
            prediction_data = {
                "school_id": school_id,
                "code": code,
                "year": year,
                "province": school_info["province"],
                "type": school_info["type"],
                "is_985": school_info["is_985"],
                "is_211": school_info["is_211"],
                "rank": school_info["rank"],
                "applicants": applicants
            }
            
            # 创建DataFrame
            predict_df = self.spark.createDataFrame([prediction_data])
            
            # 应用特征工程
            # 字符串索引转换
            string_indexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="keep")
            predict_df = string_indexer.fit(predict_df).transform(predict_df)
            
            # 对专业代码进行索引
            major_indexer = StringIndexer(inputCol="code", outputCol="code_index", handleInvalid="keep")
            predict_df = major_indexer.fit(predict_df).transform(predict_df)
            
            # 对省份进行索引
            province_indexer = StringIndexer(inputCol="province", outputCol="province_index", handleInvalid="keep")
            predict_df = province_indexer.fit(predict_df).transform(predict_df)
            
            # 确定特征列
            original_feature_cols = meta.get('original_feature_columns', ["applicants", "is_985", "is_211", 
                                                                         "type_index", "code_index", 
                                                                         "province_index", "rank", "year"])
            
            # 确保所有特征列都转换为Double类型
            for col_name in original_feature_cols:
                predict_df = predict_df.withColumn(f"{col_name}_double", col(col_name).cast(DoubleType()))
            
            # 获取实际使用的特征列名
            feature_cols_double = meta.get('feature_columns', [f"{col_name}_double" for col_name in original_feature_cols])
            
            # 创建特征向量
            assembler = VectorAssembler(
                inputCols=feature_cols_double, 
                outputCol="features",
                handleInvalid="skip"
            )
            predict_df = assembler.transform(predict_df)
            
            # 标准化特征
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(predict_df)
            predict_df = scaler_model.transform(predict_df)
            
            # 预测
            predictions = model.transform(predict_df)
            predicted_score = float(predictions.select("prediction").first()[0])
            
            logger.info(f"预测完成！学校(ID:{school_id})专业(code:{code})在{year}年的预测{target}为: {predicted_score:.2f}")
            return predicted_score
            
        except Exception as e:
            logger.error(f"预测录取分数线失败: {e}")
            raise
    
    def run_analysis(self):
        """运行完整的数据分析流程"""
        try:
            logger.info("开始运行完整数据分析流程...")
            
            # 1. 加载数据
            data_dict = self.load_data()
            
            # 2. 专业热度分析
            top_majors = self.analyze_major_popularity(data_dict)
            
            # 3. 录取趋势分析
            yearly_trends = self.analyze_admission_trends(data_dict)
            
            # 4. 分数分布分析
            score_distributions = self.analyze_score_distribution(data_dict)
            
            # 5. 训练分数预测模型
            score_model = self.train_score_prediction_model(data_dict)
            
            # 6. 训练录取概率模型
            admission_model = self.train_admission_probability_model(data_dict)
            
            logger.info("数据分析流程完成")
            
            # 返回分析结果
            return {
                'top_majors': top_majors,
                'yearly_trends': yearly_trends,
                'score_distributions': score_distributions,
                'score_model': score_model,
                'admission_model': admission_model
            }
            
        except Exception as e:
            logger.error(f"运行数据分析流程失败: {e}")
            raise
        finally:
            # 关闭Spark会话
            if hasattr(self, 'spark'):
                self.spark.stop()


if __name__ == "__main__":
    print("开始使用Spark进行考研数据分析和机器学习预测...")
    
    analyzer = KaoyanDataAnalyzer()
    results = analyzer.run_analysis()
    
    print("数据分析和机器学习预测完成！结果已保存至analysis目录。") 