#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import pickle
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
import pandas as pd
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler

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
        logging.FileHandler(os.path.join(logs_dir, 'admission_predictor.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AdmissionPredictor')

class KaoyanAdmissionPredictor:
    """考研录取概率预测工具"""
    
    def __init__(self):
        """初始化预测器"""
        # 数据目录
        self.data_dir = os.path.join(PROJECT_ROOT, 'data')
        self.clean_data_dir = os.path.join(self.data_dir, 'clean')
        self.models_dir = os.path.join(PROJECT_ROOT, 'models')
        
        # 创建Spark会话
        self.spark = self._create_spark_session()
        
        # 加载参考数据
        self._load_reference_data()
        
        logger.info("录取概率预测器已准备就绪")
    
    def _create_spark_session(self):
        """创建Spark会话"""
        try:
            # 创建Spark会话
            spark = SparkSession.builder \
                .appName("KaoyanAdmissionPredictor") \
                .config("spark.executor.memory", "1g") \
                .config("spark.driver.memory", "1g") \
                .getOrCreate()
            
            # 设置日志级别为ERROR，减少控制台输出
            spark.sparkContext.setLogLevel("ERROR")
            
            logger.info("成功创建Spark会话")
            return spark
        except Exception as e:
            logger.error(f"创建Spark会话失败: {e}")
            raise
    
    def _load_reference_data(self):
        """加载参考数据"""
        try:
            logger.info("加载参考数据...")
            
            # 加载学校数据
            schools_path = os.path.join(self.clean_data_dir, 'schools.csv')
            if os.path.exists(schools_path):
                self.schools_df = self.spark.read.csv(schools_path, header=True, inferSchema=True)
                logger.info(f"学校数据加载完成: {self.schools_df.count()} 条记录")
            else:
                logger.warning(f"学校数据文件不存在: {schools_path}")
                self.schools_df = None
            
            # 加载专业数据
            majors_path = os.path.join(self.clean_data_dir, 'majors.csv')
            if os.path.exists(majors_path):
                self.majors_df = self.spark.read.csv(majors_path, header=True, inferSchema=True)
                logger.info(f"专业数据加载完成: {self.majors_df.count()} 条记录")
            else:
                logger.warning(f"专业数据文件不存在: {majors_path}")
                self.majors_df = None
        
        except Exception as e:
            logger.error(f"加载参考数据失败: {e}")
            raise
    
    def _load_model(self, model_type='admission_probability'):
        """加载模型
        
        参数:
            model_type: 模型类型，'admission_probability' 或 'score_prediction'
        
        返回:
            模型字典，包含模型和元数据
        """
        try:
            logger.info(f"加载{model_type}模型...")
            
            # 查找可用的模型
            model_names = ['LogisticRegression', 'RandomForest', 'GBT', 'LinearRegression']
            available_models = {}
            
            for name in model_names:
                model_path = os.path.join(self.models_dir, f"{model_type}_{name}")
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    available_models[name] = model_path
            
            if not available_models:
                raise ValueError(f"未找到任何{model_type}模型，请先运行模型训练脚本")
            
            # 选择第一个可用的模型
            model_name = next(iter(available_models))
            model_path = available_models[model_name]
            
            # 根据模型类型和名称加载模型
            if model_type == 'admission_probability':
                if model_name == 'LogisticRegression':
                    model = LogisticRegressionModel.load(model_path)
                elif model_name == 'RandomForest':
                    model = RandomForestClassificationModel.load(model_path)
                elif model_name == 'GBT':
                    model = GBTClassificationModel.load(model_path)
                else:
                    raise ValueError(f"不支持的模型类型: {model_name}")
            else:  # score_prediction
                if model_name == 'LinearRegression':
                    model = LinearRegressionModel.load(model_path)
                elif model_name == 'RandomForest':
                    model = RandomForestRegressionModel.load(model_path)
                elif model_name == 'GBT':
                    model = GBTRegressionModel.load(model_path)
                else:
                    raise ValueError(f"不支持的模型类型: {model_name}")
            
            # 元数据路径
            meta_path = os.path.join(self.models_dir, f"{model_type}_{model_name}_meta.json")
            meta = {}
            
            # 如果存在元数据，则加载
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                logger.warning(f"未找到模型元数据: {meta_path}，使用默认值")
                # 使用默认特征列
                meta = {
                    'feature_columns': ["politics", "english", "special_one", "special_two", 
                                       "is_985", "is_211", "type_index", "code_index", "rank", "year"]
                }
            
            logger.info(f"模型加载完成: {model_name}")
            
            return {
                'model': model,
                'model_name': model_name,
                'meta': meta
            }
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def search_school(self, keyword):
        """搜索学校信息"""
        try:
            if self.schools_df is None:
                logger.error("学校数据未加载，无法搜索")
                return []
            
            # 使用关键字搜索学校名称
            results = self.schools_df.filter(
                col("school_name").contains(keyword)
            ).collect()
            
            if not results:
                logger.info(f"未找到匹配'{keyword}'的学校")
                return []
            
            # 转换为字典列表
            school_list = []
            for row in results:
                school_dict = row.asDict()
                school_list.append(school_dict)
            
            logger.info(f"找到 {len(school_list)} 所匹配'{keyword}'的学校")
            return school_list
            
        except Exception as e:
            logger.error(f"搜索学校失败: {e}")
            raise
    
    def search_major(self, keyword):
        """搜索专业信息"""
        try:
            if self.majors_df is None:
                logger.error("专业数据未加载，无法搜索")
                return []
            
            # 使用关键字搜索专业名称
            results = self.majors_df.filter(
                col("name").contains(keyword) | col("code").contains(keyword)
            ).collect()
            
            if not results:
                logger.info(f"未找到匹配'{keyword}'的专业")
                return []
            
            # 转换为字典列表
            major_list = []
            for row in results:
                major_dict = row.asDict()
                major_list.append(major_dict)
            
            logger.info(f"找到 {len(major_list)} 个匹配'{keyword}'的专业")
            return major_list
            
        except Exception as e:
            logger.error(f"搜索专业失败: {e}")
            raise
    
    def predict(self, student_data, model_type='admission_probability'):
        """预测录取概率或分数
        
        参数:
            student_data: 学生数据字典
            model_type: 预测类型，'admission_probability' 或 'score_prediction'
        
        返回:
            预测结果
        """
        try:
            logger.info(f"开始{model_type}预测...")
            
            # 加载模型
            model_dict = self._load_model(model_type)
            model = model_dict['model']
            meta = model_dict['meta']
            
            # 创建单行DataFrame
            student_row = self.spark.createDataFrame([student_data])
            
            # 获取学校信息
            if 'school_id' in student_data:
                school_info = self.schools_df.filter(col("school_id") == student_data['school_id'])
                
                if school_info.count() == 0:
                    raise ValueError(f"未找到ID为{student_data['school_id']}的学校信息")
                
                school_row = school_info.first()
                
                # 补充学校信息
                student_row = student_row.withColumn("is_985", lit(school_row["is_985"])) \
                                        .withColumn("is_211", lit(school_row["is_211"])) \
                                        .withColumn("type", lit(school_row["type"])) \
                                        .withColumn("rank", lit(school_row["rank"]))
            
            # 处理缺失值
            student_row = student_row.na.fill({
                "politics": 0,
                "english": 0,
                "special_one": 0,
                "special_two": 0,
                "rank": 999,
                "is_985": 0,
                "is_211": 0,
                "year": 2022
            })
            
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
            if model_type == 'admission_probability':
                if 'probability' in prediction.columns:
                    result = prediction.select("probability").first()[0][1]  # 第二个元素是正类的概率
                else:
                    # 如果没有概率列，则使用原始预测
                    result = float(prediction.select("prediction").first()[0])
                logger.info(f"预测完成，录取概率: {result:.4f}")
            else:
                # 分数预测
                result = float(prediction.select("prediction").first()[0])
                logger.info(f"预测完成，预测分数: {result:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def predict_admission_threshold(self, school_id, code, year=2023, target="min_score"):
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
                model = LinearRegressionModel.load(model_path)
            elif model_name == 'RandomForest':
                model = RandomForestRegressionModel.load(model_path)
            elif model_name == 'GBT':
                model = GBTRegressionModel.load(model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
            
            # 获取学校信息
            if self.schools_df is None:
                raise ValueError("学校数据未加载，无法预测")
                
            school_info = self.schools_df.filter(col("school_id") == school_id).first()
            
            if not school_info:
                raise ValueError(f"未找到ID为{school_id}的学校信息")
            
            # 估算申请人数（从scores表中获取）
            # 在真实系统中，这里应该查询分数数据表获取历史申请人数
            # 这里我们使用一个默认值
            applicants = 20
            
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
            
            # 如果是分数，确保结果为整数并在合理范围内
            if target.endswith("_score"):
                predicted_score = max(0, min(500, round(predicted_score)))
            
            logger.info(f"预测完成！学校(ID:{school_id})专业(code:{code})在{year}年的预测{target}为: {predicted_score}")
            return predicted_score
            
        except Exception as e:
            logger.error(f"预测录取分数线失败: {e}")
            raise
        
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='考研录取预测工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 查找学校命令
    search_school_parser = subparsers.add_parser('search_school', help='查找学校信息')
    search_school_parser.add_argument('keyword', type=str, help='学校名称关键词')
    
    # 查找专业命令
    search_major_parser = subparsers.add_parser('search_major', help='查找专业信息')
    search_major_parser.add_argument('keyword', type=str, help='专业名称关键词')
    
    # 预测录取概率命令
    predict_prob_parser = subparsers.add_parser('predict_prob', help='预测录取概率')
    predict_prob_parser.add_argument('school_id', type=int, help='学校ID')
    predict_prob_parser.add_argument('major_code', type=str, help='专业代码')
    predict_prob_parser.add_argument('score', type=int, help='考生分数')
    predict_prob_parser.add_argument('--year', type=int, default=2023, help='预测年份，默认2023')
    
    # 预测分数线命令
    predict_score_parser = subparsers.add_parser('predict_score', help='预测录取分数线')
    predict_score_parser.add_argument('school_id', type=int, help='学校ID')
    predict_score_parser.add_argument('major_code', type=str, help='专业代码')
    predict_score_parser.add_argument('--year', type=int, default=2023, help='预测年份，默认2023')
    predict_score_parser.add_argument('--target', type=str, choices=['min_score', 'avg_score'], 
                                     default='min_score', help='预测目标，最低分或平均分，默认最低分')
    
    args = parser.parse_args()
    
    try:
        # 创建预测器实例
        predictor = KaoyanAdmissionPredictor()
        
        if args.command == 'search_school':
            # 查找学校
            schools = predictor.search_school(args.keyword)
            if schools:
                print("\n找到以下学校信息：")
                for school in schools:
                    print(f"ID: {school['school_id']}, 名称: {school['school_name']}, "
                          f"省份: {school['province']}, 类型: {school['type']}, "
                          f"是否985: {'是' if school['is_985'] else '否'}, "
                          f"是否211: {'是' if school['is_211'] else '否'}, "
                          f"排名: {school['rank']}")
            else:
                print(f"未找到包含关键词 '{args.keyword}' 的学校")
                
        elif args.command == 'search_major':
            # 查找专业
            majors = predictor.search_major(args.keyword)
            if majors:
                print("\n找到以下专业信息：")
                for major in majors:
                    print(f"代码: {major['code']}, 名称: {major['name']}, "
                          f"学科门类: {major['discipline']}, 一级学科: {major['category']}")
            else:
                print(f"未找到包含关键词 '{args.keyword}' 的专业")
                
        elif args.command == 'predict_prob':
            # 准备学生数据
            student_data = {
                'school_id': args.school_id,
                'code': args.major_code,
                'score': args.score,
                'year': args.year
            }
            
            # 预测录取概率
            probability = predictor.predict_admission_probability({}, student_data)
            print(f"\n录取预测结果：")
            print(f"学校ID: {args.school_id}, 专业代码: {args.major_code}")
            print(f"考生分数: {args.score}, 预测年份: {args.year}")
            print(f"录取概率: {probability:.2%}")
            
            # 添加建议
            if probability >= 0.8:
                print("建议: 录取可能性很高，可以作为理想院校报考")
            elif probability >= 0.5:
                print("建议: 录取可能性较大，可以作为较理想院校报考")
            elif probability >= 0.3:
                print("建议: 录取可能性存在，但有一定风险，建议作为冲刺院校")
            else:
                print("建议: 录取可能性较低，建议考虑其他院校或提高分数")
                
        elif args.command == 'predict_score':
            # 预测录取分数线
            predicted_score = predictor.predict_admission_threshold(
                args.school_id, 
                args.major_code, 
                args.year, 
                args.target
            )
            
            score_type = "最低录取分数线" if args.target == "min_score" else "平均录取分数线"
            print(f"\n分数线预测结果：")
            print(f"学校ID: {args.school_id}, 专业代码: {args.major_code}")
            print(f"预测年份: {args.year}, 预测类型: {score_type}")
            print(f"预测分数线: {predicted_score}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"错误: {str(e)}")
        logger.error(f"运行错误: {str(e)}")
    finally:
        # 确保Spark会话关闭
        if 'predictor' in locals() and hasattr(predictor, 'spark'):
            try:
                predictor.spark.stop()
                logger.info("Spark会话已关闭")
            except:
                pass

if __name__ == "__main__":
    main() 