#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
import logging
from pathlib import Path

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'data_service.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataService')

class KaoyanDataService:
    """读取分析结果数据服务"""
    
    def __init__(self):
        """初始化数据服务"""
        # 数据目录
        self.data_dir = os.path.join(PROJECT_ROOT, 'data')
        self.clean_data_dir = os.path.join(self.data_dir, 'clean')
        self.analysis_dir = os.path.join(self.data_dir, 'analysis')
        self.cache_dir = os.path.join(PROJECT_ROOT, 'app', 'static', 'cache')
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 数据缓存
        self._schools_df = None
        self._majors_df = None
        self._province_scores = None
        
        logger.info("数据服务已准备就绪")
    
    def _read_csv_dir(self, directory):
        """读取CSV文件或目录"""
        try:
            if os.path.isdir(directory):
                # 如果是目录，查找目录下的CSV文件
                csv_files = [f for f in os.listdir(directory) 
                             if f.endswith('.csv') and not f.startswith('.')]
                
                if not csv_files:
                    # 尝试查找part文件
                    part_files = [f for f in os.listdir(directory) 
                                if f.startswith('part-') and f.endswith('.csv')]
                    
                    if part_files:
                        # 读取第一个part文件
                        return pd.read_csv(os.path.join(directory, part_files[0]), 
                                        encoding='utf-8')
                else:
                    # 读取第一个CSV文件
                    return pd.read_csv(os.path.join(directory, csv_files[0]), 
                                      encoding='utf-8')
            else:
                # 如果是文件，直接读取
                return pd.read_csv(directory, encoding='utf-8')
        except Exception as e:
            logger.error(f"读取CSV文件失败: {directory}, 错误: {e}")
            return pd.DataFrame()  # 返回空DataFrame
    
    def get_schools(self):
        """获取学校数据"""
        if self._schools_df is None:
            schools_path = os.path.join(self.clean_data_dir, 'schools.csv')
            self._schools_df = pd.read_csv(schools_path, encoding='utf-8')
        return self._schools_df
    
    def get_majors(self):
        """获取专业数据"""
        if self._majors_df is None:
            majors_path = os.path.join(self.clean_data_dir, 'majors.csv')
            self._majors_df = pd.read_csv(majors_path, encoding='utf-8')
        return self._majors_df
    
    def get_top_schools(self, limit=20):
        """获取热门学校排名"""
        try:
            # 检查缓存
            cache_file = os.path.join(self.cache_dir, 'top_schools.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 读取分数数据
            scores_dir = os.path.join(self.clean_data_dir, 'scores.csv')
            scores_df = self._read_csv_dir(scores_dir)
            
            # 获取学校信息
            schools_df = self.get_schools()
            
            # 计算每个学校的申请人数
            school_counts = scores_df.groupby('school_id').size().reset_index(name='count')
            
            # 与学校信息合并
            result_df = school_counts.merge(schools_df, on='school_id', how='left')
            
            # 按申请人数排序
            top_schools = result_df.sort_values('count', ascending=False).head(limit)
            
            # 选择需要的列
            top_schools = top_schools[['school_id', 'school_name', 'count', 'province', 'type', 'is_985', 'is_211']]
            
            # 转换为列表字典
            result = top_schools.to_dict('records')
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            
            return result
        except Exception as e:
            logger.error(f"获取热门学校排名失败: {e}")
            return []
    
    def get_top_majors(self, limit=20):
        """获取热门专业排名"""
        try:
            # 检查缓存
            cache_file = os.path.join(self.cache_dir, 'top_majors.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 读取分数数据
            scores_dir = os.path.join(self.clean_data_dir, 'scores.csv')
            scores_df = self._read_csv_dir(scores_dir)
            
            # 获取专业信息
            majors_df = self.get_majors()
            
            # 计算每个专业的申请人数
            major_counts = scores_df.groupby(['code', 'name']).size().reset_index(name='count')
            
            # 与专业信息合并
            result_df = major_counts.merge(majors_df, on='code', how='left')
            
            # 按申请人数排序
            top_majors = result_df.sort_values('count', ascending=False).head(limit)
            
            # 选择需要的列并处理列名重复问题
            top_majors = top_majors[['code', 'name_x', 'count', 'category']]
            top_majors = top_majors.rename(columns={'name_x': 'name'})
            
            # 转换为列表字典
            result = top_majors.to_dict('records')
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            
            return result
        except Exception as e:
            logger.error(f"获取热门专业排名失败: {e}")
            return []
    
    def get_top_provinces(self, limit=20):
        """获取热门省份排名"""
        try:
            # 检查缓存
            cache_file = os.path.join(self.cache_dir, 'top_provinces.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 如果省份分数分布已缓存，直接使用
            if self._province_scores is not None:
                top_provinces = self._province_scores.head(limit)
                result = top_provinces.to_dict('records')
                return result
            
            # 读取分数数据
            scores_dir = os.path.join(self.clean_data_dir, 'scores.csv')
            scores_df = self._read_csv_dir(scores_dir)
            
            # 获取学校信息
            schools_df = self.get_schools()
            
            # 合并数据
            joined_df = scores_df.merge(
                schools_df[['school_id', 'province']],
                on='school_id',
                how='left'
            )
            
            # 计算每个省份的申请人数
            province_counts = joined_df.groupby('province').size().reset_index(name='count')
            
            # 按申请人数排序
            self._province_scores = province_counts.sort_values('count', ascending=False)
            
            # 获取前N个
            top_provinces = self._province_scores.head(limit)
            
            # 转换为列表字典
            result = top_provinces.to_dict('records')
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            
            return result
        except Exception as e:
            logger.error(f"获取热门省份排名失败: {e}")
            # 返回模拟数据作为备选
            return self.get_mock_province_data()
    
    def get_mock_province_data(self):
        """返回模拟的省份数据"""
        mock_data = [
            {"province": "北京", "count": 12686},
            {"province": "江苏", "count": 9609},
            {"province": "湖北", "count": 7279},
            {"province": "山东", "count": 5900},
            {"province": "辽宁", "count": 5846},
            {"province": "上海", "count": 5450},
            {"province": "广东", "count": 5336},
            {"province": "陕西", "count": 4945},
            {"province": "河南", "count": 4600},
            {"province": "浙江", "count": 4555},
            {"province": "四川", "count": 3900},
            {"province": "安徽", "count": 3800},
            {"province": "天津", "count": 3700},
            {"province": "吉林", "count": 3500},
            {"province": "湖南", "count": 3200},
            {"province": "黑龙江", "count": 3100},
            {"province": "重庆", "count": 2800},
            {"province": "福建", "count": 2500},
            {"province": "甘肃", "count": 2200},
            {"province": "贵州", "count": 2000}
        ]
        return mock_data
    
    def get_all_stats(self):
        """获取所有统计数据"""
        try:
            # 获取各个统计结果
            top_schools = self.get_top_schools()
            top_majors = self.get_top_majors()
            top_provinces = self.get_top_provinces()
            
            # 组合结果
            result = {
                'top_schools': top_schools,
                'top_majors': top_majors,
                'top_provinces': top_provinces
            }
            
            return result
        except Exception as e:
            logger.error(f"获取所有统计数据失败: {e}")
            return {
                'top_schools': [],
                'top_majors': [],
                'top_provinces': []
            } 