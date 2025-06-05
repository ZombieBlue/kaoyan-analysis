#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import configparser
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, trim, regexp_replace, year, avg, count
from pyspark.sql.types import IntegerType, FloatType, StringType
import pandas as pd
from pathlib import Path
import argparse

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
        logging.FileHandler(os.path.join(logs_dir, 'data_cleaner.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataCleaner')

class DataCleaner:
    """从MySQL数据库中读取考研数据，使用Spark进行数据清洗和转换"""
    
    def __init__(self, config_path=None):
        """初始化数据清洗器"""
        # 如果没有提供配置路径，使用默认路径（与数据库初始化使用同一个配置文件）
        if config_path is None:
            self.config_path = os.path.join(PROJECT_ROOT, 'config', 'database.ini')
        else:
            self.config_path = config_path
            
        # 数据目录
        self.data_dir = os.path.join(PROJECT_ROOT, 'data')
        self.clean_data_dir = os.path.join(self.data_dir, 'clean')
        
        # 确保目录存在
        os.makedirs(self.clean_data_dir, exist_ok=True)
        
        # 加载数据库配置（虽然本脚本不直接连接数据库，但与数据库初始化使用同一配置文件保持一致）
        self.db_config = self._load_config()
        
        # 创建Spark会话
        self.spark = self._create_spark_session()
        
        logger.info("数据清洗器已准备就绪")
    
    def _load_config(self):
        """加载数据库配置"""
        config = configparser.ConfigParser()
        config.read(self.config_path, encoding='utf-8')
        
        if 'mysql' in config:
            db_config = {
                'host': config['mysql']['host'],
                'database': config['mysql']['database'],
                'user': config['mysql']['user'],
                'password': config['mysql']['password'],
                'port': config['mysql'].getint('port', 3306)
            }
            return db_config
        else:
            logger.error("配置文件中没有找到MySQL配置部分")
            raise ValueError("配置文件中没有找到MySQL配置部分")
    
    def _create_spark_session(self):
        """创建Spark会话"""
        try:
            # 创建Spark会话 - 移除依赖下载方式，改用本地JAR文件
            spark = SparkSession.builder \
                .appName("KaoyanDataCleaner") \
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
    
    def load_data_from_mysql(self):
        """从CSV文件加载数据（方法名保持不变以保持代码结构一致）
        
        注意：虽然方法名包含MySQL，但实际上是从CSV文件读取数据，因为直接JDBC连接存在依赖问题。
        如果需要从MySQL直接读取，需要解决Spark的MySQL连接器依赖问题。
        """
        try:
            raw_data_dir = os.path.join(self.data_dir, 'raw')
            
            # 找到最新的CSV文件
            def get_latest_file(prefix):
                files = [f for f in os.listdir(raw_data_dir) if f.startswith(prefix) and f.endswith('.csv')]
                files.sort(reverse=True)
                return files[0] if files else None
            
            # 获取最新的学校数据和分数数据文件
            school_file = get_latest_file('school_list_')
            score_file = get_latest_file('admission_scores_')
            
            if not school_file or not score_file:
                raise FileNotFoundError("未找到原始CSV数据文件")
            
            logger.info(f"找到学校数据文件: {school_file}")
            logger.info(f"找到分数数据文件: {score_file}")
            
            # 从CSV读取数据
            # 读取学校数据
            logger.info("读取学校数据...")
            schools_df = self.spark.read.option("header", "true").option("encoding", "UTF-8").csv(os.path.join(raw_data_dir, school_file))
            
            # 读取分数数据
            logger.info("读取录取分数数据...")
            scores_df = self.spark.read.option("header", "true").option("encoding", "UTF-8").csv(os.path.join(raw_data_dir, score_file))
            
            # 创建专业数据 - 从分数数据中提取唯一专业代码和名称
            logger.info("从分数数据中提取专业信息...")
            majors_df = scores_df.select("code", "name").dropDuplicates()
            # 添加category列
            majors_df = majors_df.withColumn("category", lit("未分类"))
            
            logger.info(f"数据加载完成: 学校 {schools_df.count()} 条, 分数 {scores_df.count()} 条, 专业 {majors_df.count()} 条")
            
            return {
                'schools': schools_df,
                'scores': scores_df,
                'majors': majors_df
            }
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def clean_schools_data(self, schools_df):
        """清洗学校数据"""
        try:
            logger.info("开始清洗学校数据...")
            
            # 1. 去除重复记录
            schools_df = schools_df.dropDuplicates(['school_id'])
            
            # 2. 数据类型转换和标准化
            schools_df = schools_df.withColumn("school_id", col("school_id").cast(IntegerType()))
            schools_df = schools_df.withColumn("school_name", trim(col("school_name")))
            schools_df = schools_df.withColumn("province", trim(col("province")))
            schools_df = schools_df.withColumn("type", trim(col("type")))
            
            # 3. 处理缺失值
            schools_df = schools_df.withColumn("recruit_number", 
                when(col("recruit_number").isNull(), 0).otherwise(col("recruit_number")))
            schools_df = schools_df.withColumn("rank", 
                when(col("rank").isNull(), 9999).otherwise(col("rank")))
            
            # 4. 标准化布尔值
            schools_df = schools_df.withColumn("is_985", 
                when(col("is_985") == "是", 1).otherwise(0))
            schools_df = schools_df.withColumn("is_211", 
                when(col("is_211") == "是", 1).otherwise(0))
            
            logger.info(f"学校数据清洗完成, 剩余 {schools_df.count()} 条记录")
            return schools_df
        
        except Exception as e:
            logger.error(f"清洗学校数据失败: {e}")
            raise
    
    def clean_scores_data(self, scores_df):
        """清洗录取分数数据"""
        try:
            logger.info("开始清洗录取分数数据...")
            
            # 1. 去除重复记录
            scores_df = scores_df.dropDuplicates(['id'])
            
            # 2. 数据类型转换
            scores_df = scores_df.withColumn("school_id", col("school_id").cast(IntegerType()))
            scores_df = scores_df.withColumn("year", col("year").cast(IntegerType()))
            scores_df = scores_df.withColumn("crawl_year", col("crawl_year").cast(IntegerType()))
            
            # 3. 处理缺失值
            # 使用0替代缺失的分数
            numeric_cols = ["politics", "english", "special_one", "special_two", "total",
                           "diff_total", "diff_politics", "diff_english", "diff_special_one", "diff_special_two"]
            
            for col_name in numeric_cols:
                scores_df = scores_df.withColumn(col_name, 
                    when(col(col_name).isNull(), 0).otherwise(col(col_name)))
            
            # 4. 去除无效数据
            # 过滤掉总分为0的记录
            scores_df = scores_df.filter(col("total") > 0)
            
            # 5. 清理文本字段
            string_cols = ["school_name", "depart_name", "code", "name", "politics_str", 
                          "english_str", "special_one_str", "special_two_str"]
            
            for col_name in string_cols:
                scores_df = scores_df.withColumn(col_name, 
                    when(col(col_name).isNull(), "").otherwise(trim(col(col_name))))
            
            logger.info(f"录取分数数据清洗完成, 剩余 {scores_df.count()} 条记录")
            return scores_df
        
        except Exception as e:
            logger.error(f"清洗录取分数数据失败: {e}")
            raise
    
    def clean_majors_data(self, majors_df):
        """清洗专业数据"""
        try:
            logger.info("开始清洗专业数据...")
            
            # 1. 去除重复记录
            majors_df = majors_df.dropDuplicates(['code'])
            
            # 2. 数据标准化
            majors_df = majors_df.withColumn("code", trim(col("code")))
            majors_df = majors_df.withColumn("name", trim(col("name")))
            
            # 3. 处理缺失值
            majors_df = majors_df.withColumn("category", 
                when(col("category").isNull(), "其他").otherwise(col("category")))
            
            # 4. 过滤无效数据
            majors_df = majors_df.filter(col("code").isNotNull() & col("name").isNotNull())
            
            logger.info(f"专业数据清洗完成, 剩余 {majors_df.count()} 条记录")
            return majors_df
        
        except Exception as e:
            logger.error(f"清洗专业数据失败: {e}")
            raise
    
    def create_analytical_views(self, data_dict):
        """创建分析用视图"""
        try:
            logger.info("开始创建分析用视图...")
            
            schools_df = data_dict['schools']
            scores_df = data_dict['scores']
            majors_df = data_dict['majors']
            
            # 1. 创建学校录取分数汇总视图
            logger.info("创建学校录取分数汇总视图...")
            school_scores_summary = scores_df.groupBy("school_id", "school_name", "year") \
                .agg(
                    avg("total").alias("avg_total_score"),
                    avg("politics").alias("avg_politics_score"),
                    avg("english").alias("avg_english_score"),
                    avg("special_one").alias("avg_special_one_score"),
                    avg("special_two").alias("avg_special_two_score"),
                    count("*").alias("major_count")
                )
            
            # 2. 与学校信息关联
            school_scores_summary = school_scores_summary.join(
                schools_df.select("school_id", "province", "type", "is_985", "is_211", "rank"),
                on="school_id",
                how="left"
            )
            
            # 3. 创建专业录取分数视图
            logger.info("创建专业录取分数视图...")
            major_scores = scores_df.join(
                majors_df.select("code", "name", "category"),
                on=[scores_df.code == majors_df.code, scores_df.name == majors_df.name],
                how="left"
            ).select(
                scores_df["*"],
                majors_df["category"].alias("major_category")
            )
            
            # 4. 按专业统计平均分
            major_avg_scores = scores_df.groupBy("code", "name", "year") \
                .agg(
                    avg("total").alias("avg_total_score"),
                    count("*").alias("school_count")
                )
            
            # 5. 985/211院校与普通院校对比视图
            logger.info("创建985/211院校与普通院校对比视图...")
            elite_vs_normal = scores_df.join(
                schools_df.select("school_id", "is_985", "is_211"),
                on="school_id",
                how="left"
            ).withColumn(
                "school_type",
                when((col("is_985") == 1) & (col("is_211") == 1), "985&211")
                .when((col("is_985") == 0) & (col("is_211") == 1), "211")
                .otherwise("普通院校")
            ).groupBy("school_type", "year") \
                .agg(
                    avg("total").alias("avg_total_score"),
                    count("*").alias("record_count")
                )
            
            data_dict.update({
                'school_scores_summary': school_scores_summary,
                'major_scores': major_scores,
                'major_avg_scores': major_avg_scores,
                'elite_vs_normal': elite_vs_normal
            })
            
            logger.info("分析用视图创建完成")
            return data_dict
        
        except Exception as e:
            logger.error(f"创建分析用视图失败: {e}")
            raise
    
    def save_to_csv(self, data_dict):
        """将数据保存为CSV格式"""
        try:
            logger.info("开始将清洗后的数据保存为CSV...")
            
            # 保存所有数据框为CSV
            for name, df in data_dict.items():
                output_path = os.path.join(self.clean_data_dir, f"{name}.csv")
                
                logger.info(f"保存 {name} 到 {output_path}")
                
                # 使用pandas保存小数据集，保持中文显示正常
                if name in ['schools', 'majors']:
                    # 转换为pandas DataFrame
                    pandas_df = df.toPandas()
                    # 保存为CSV，确保中文正确显示
                    pandas_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                else:
                    # 使用Spark直接保存大数据集
                    df.coalesce(1).write.option("header", "true").option("encoding", "UTF-8").mode("overwrite").csv(output_path)
            
            logger.info("所有数据已保存为CSV格式")
            
            # 返回保存的文件路径列表
            return self.clean_data_dir
        
        except Exception as e:
            logger.error(f"保存CSV数据失败: {e}")
            raise
    
    def load_directly_from_mysql(self):
        """直接从MySQL数据库加载数据（需要解决依赖问题才能使用）
        
        此方法要求正确配置MySQL JDBC连接器。使用前需要确保:
        1. 下载MySQL Connector/J JAR文件
        2. 将JAR文件放在Spark的classpath中，或使用--jars参数指定
        
        例如:
        spark-submit --jars /path/to/mysql-connector-java-8.0.28.jar data_cleaner.py
        """
        try:
            # 构建JDBC URL
            jdbc_url = f"jdbc:mysql://{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}?useUnicode=true&characterEncoding=utf8"
            
            # 数据库连接属性
            connection_properties = {
                "user": self.db_config['user'],
                "password": self.db_config['password'],
                "driver": "com.mysql.cj.jdbc.Driver"
            }
            
            # 加载学校数据
            logger.info("从MySQL加载学校数据...")
            schools_df = self.spark.read.jdbc(
                url=jdbc_url,
                table="schools",
                properties=connection_properties
            )
            
            # 加载录取分数数据
            logger.info("从MySQL加载录取分数数据...")
            scores_df = self.spark.read.jdbc(
                url=jdbc_url,
                table="admission_scores",
                properties=connection_properties
            )
            
            # 加载专业数据
            logger.info("从MySQL加载专业数据...")
            majors_df = self.spark.read.jdbc(
                url=jdbc_url,
                table="majors",
                properties=connection_properties
            )
            
            logger.info(f"数据加载完成: 学校 {schools_df.count()} 条, 分数 {scores_df.count()} 条, 专业 {majors_df.count()} 条")
            
            return {
                'schools': schools_df,
                'scores': scores_df,
                'majors': majors_df
            }
        except Exception as e:
            logger.error(f"从MySQL加载数据失败: {e}")
            logger.info("尝试从CSV文件加载数据...")
            return self.load_data_from_mysql()  # 回退到从CSV读取
            
    def clean_and_transform(self, use_mysql_directly=False):
        """执行数据清洗和转换流程
        
        参数:
            use_mysql_directly: 是否直接从MySQL加载数据，默认为False
        """
        try:
            # 1. 加载数据
            if use_mysql_directly:
                # 尝试直接从MySQL加载，如果失败则回退到CSV
                try:
                    logger.info("尝试直接从MySQL数据库加载数据...")
                    data_dict = self.load_directly_from_mysql()
                except Exception as e:
                    logger.error(f"直接从MySQL加载失败，回退到CSV: {e}")
                    data_dict = self.load_data_from_mysql()
            else:
                # 从CSV文件加载数据
                data_dict = self.load_data_from_mysql()
            
            # 2. 清洗每个数据集
            data_dict['schools'] = self.clean_schools_data(data_dict['schools'])
            data_dict['scores'] = self.clean_scores_data(data_dict['scores'])
            data_dict['majors'] = self.clean_majors_data(data_dict['majors'])
            
            # 3. 创建分析用视图
            data_dict = self.create_analytical_views(data_dict)
            
            # 4. 保存为CSV格式
            output_dir = self.save_to_csv(data_dict)
            
            logger.info(f"数据清洗和转换完成，结果已保存至: {output_dir}")
            
            # 5. 关闭Spark会话
            self.spark.stop()
            
            return True
        
        except Exception as e:
            logger.error(f"数据清洗和转换过程出错: {e}")
            # 关闭Spark会话
            if hasattr(self, 'spark'):
                self.spark.stop()
            return False


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='考研数据清洗和转换工具')
    parser.add_argument('--use-mysql', action='store_true', help='直接从MySQL数据库加载数据（需要解决依赖问题）')
    args = parser.parse_args()
    
    print("开始使用Spark进行考研数据清洗和转换...")
    if args.use_mysql:
        print("尝试直接从MySQL数据库加载数据...")
    else:
        print("从CSV文件加载数据...")
        
    cleaner = DataCleaner()
    success = cleaner.clean_and_transform(use_mysql_directly=args.use_mysql)
    
    if success:
        print("数据清洗和转换成功！清洗后的数据已保存为CSV格式。")
    else:
        print("数据清洗和转换失败，请查看日志获取详细信息。") 