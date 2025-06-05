#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
import configparser
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/db_init.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DatabaseInit')

class DatabaseInitializer:
    """初始化MySQL数据库，创建表结构并导入CSV数据"""
    
    def __init__(self, config_path='../config/database.ini'):
        """初始化数据库连接配置"""
        self.config_path = config_path
        self.conn = None
        self.cursor = None
        self.db_config = self._load_config()
        self.data_dir = "../data/raw"
        
        # 确保日志和配置目录存在
        os.makedirs('../logs', exist_ok=True)
        os.makedirs('../config', exist_ok=True)
        
        logger.info("数据库初始化器已准备就绪")
    
    def _load_config(self):
        """加载数据库配置"""
        config_dir = os.path.dirname(self.config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 如果配置文件不存在，创建一个默认配置
        if not os.path.exists(self.config_path):
            self._create_default_config()
            logger.info(f"已创建默认数据库配置文件: {self.config_path}")
        
        # 读取配置
        config = configparser.ConfigParser()
        config.read(self.config_path, encoding='utf-8')
        
        # 提取MySQL配置
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
            sys.exit(1)
    
    def _create_default_config(self):
        """创建默认数据库配置文件"""
        config = configparser.ConfigParser()
        config['mysql'] = {
            'host': 'localhost',
            'database': 'kaoyan_analysis',
            'user': 'root',
            'password': 'password',
            'port': '3306'
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            config.write(f)
    
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.conn = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
            self.cursor = self.conn.cursor()
            logger.info("成功连接到MySQL数据库")
            return True
        except Error as e:
            logger.error(f"连接MySQL数据库失败: {e}")
            return False
    
    def create_database(self):
        """创建数据库"""
        try:
            db_name = self.db_config['database']
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self.conn.commit()
            
            # 切换到新创建的数据库
            self.cursor.execute(f"USE {db_name}")
            logger.info(f"成功创建并使用数据库: {db_name}")
            return True
        except Error as e:
            logger.error(f"创建数据库失败: {e}")
            return False
    
    def create_tables(self):
        """创建表结构"""
        try:
            # 创建学校表 (schools)
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS schools (
                school_id INT PRIMARY KEY,
                school_name VARCHAR(100) NOT NULL,
                province VARCHAR(50),
                type VARCHAR(50),
                is_985 ENUM('是', '否') DEFAULT '否',
                is_211 ENUM('是', '否') DEFAULT '否',
                recruit_number INT,
                ranking INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            # 创建录取分数表 (admission_scores)
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS admission_scores (
                id VARCHAR(20) PRIMARY KEY,
                data_type VARCHAR(20),
                school_id INT NOT NULL,
                school_name VARCHAR(100) NOT NULL,
                depart_id INT,
                depart_name VARCHAR(100),
                code VARCHAR(20),
                name VARCHAR(100),
                politics INT,
                politics_str VARCHAR(20),
                english INT,
                english_str VARCHAR(20),
                special_one INT,
                special_one_str VARCHAR(20),
                special_two INT,
                special_two_str VARCHAR(20),
                total INT,
                note TEXT,
                year INT,
                degree_type VARCHAR(10),
                special_remark TEXT,
                diff_total INT,
                diff_politics INT,
                diff_english INT,
                diff_special_one INT,
                diff_special_two INT,
                crawl_year INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (school_id) REFERENCES schools(school_id),
                INDEX idx_school (school_id, year),
                INDEX idx_major (code, name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            # 创建学科专业表 (majors)
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS majors (
                code VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            self.conn.commit()
            logger.info("成功创建数据表结构")
            return True
        except Error as e:
            logger.error(f"创建表结构失败: {e}")
            self.conn.rollback()
            return False
    
    def get_latest_csv_files(self):
        """获取最新的CSV文件"""
        try:
            # 查找最新的学校列表文件
            school_files = [f for f in os.listdir(self.data_dir) if f.startswith('school_list_') and f.endswith('.csv')]
            school_files.sort(reverse=True)  # 按文件名排序，最新的在前面
            
            # 查找最新的分数数据文件
            score_files = [f for f in os.listdir(self.data_dir) if f.startswith('admission_scores_') and f.endswith('.csv')]
            score_files.sort(reverse=True)
            
            school_file = school_files[0] if school_files else None
            score_file = score_files[0] if score_files else None
            
            if not school_file or not score_file:
                logger.error("找不到必要的CSV文件")
                return None, None
            
            return os.path.join(self.data_dir, school_file), os.path.join(self.data_dir, score_file)
        except Exception as e:
            logger.error(f"获取最新CSV文件失败: {e}")
            return None, None
    
    def import_schools_data(self, school_file):
        """导入学校数据"""
        try:
            logger.info(f"开始导入学校数据: {school_file}")
            
            # 读取CSV文件
            schools_df = pd.read_csv(school_file, encoding='utf-8-sig')
            
            # 插入数据
            insert_query = """
            INSERT INTO schools (school_id, school_name, province, type, is_985, is_211, recruit_number, ranking)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            school_name=VALUES(school_name),
            province=VALUES(province),
            type=VALUES(type),
            is_985=VALUES(is_985),
            is_211=VALUES(is_211),
            recruit_number=VALUES(recruit_number),
            ranking=VALUES(ranking)
            """
            
            # 准备数据
            count = 0
            batch_size = 1000
            data_list = []
            
            for _, row in schools_df.iterrows():
                # 将空字符串转换为 None
                recruit_number = None if pd.isna(row['recruit_number']) or row['recruit_number'] == '' else row['recruit_number']
                
                # CSV文件中列名是rank，但数据库中列名是ranking
                ranking = None if pd.isna(row['rank']) or row['rank'] == '' else row['rank']
                
                data = (
                    int(row['school_id']),
                    row['school_name'],
                    row['province'],
                    row['type'],
                    row['is_985'],
                    row['is_211'],
                    recruit_number,
                    ranking
                )
                data_list.append(data)
                count += 1
                
                # 批量插入
                if len(data_list) >= batch_size:
                    self.cursor.executemany(insert_query, data_list)
                    self.conn.commit()
                    data_list = []
                    logger.info(f"已导入 {count} 所学校数据")
            
            # 插入剩余数据
            if data_list:
                self.cursor.executemany(insert_query, data_list)
                self.conn.commit()
            
            logger.info(f"成功导入学校数据，共 {count} 条记录")
            return count
        except Exception as e:
            logger.error(f"导入学校数据失败: {e}")
            self.conn.rollback()
            return 0
    
    def import_admission_scores(self, score_file):
        """导入录取分数数据"""
        try:
            logger.info(f"开始导入录取分数数据: {score_file}")
            
            # 由于文件可能较大，使用分块读取
            chunk_size = 10000  # 每次读取的行数
            total_count = 0
            
            # 创建一个迭代器
            chunks = pd.read_csv(score_file, encoding='utf-8-sig', chunksize=chunk_size)
            
            # 准备插入语句
            insert_query = """
            INSERT INTO admission_scores (
                id, data_type, school_id, school_name, depart_id, depart_name, 
                code, name, politics, politics_str, english, english_str, 
                special_one, special_one_str, special_two, special_two_str, 
                total, note, year, degree_type, special_remark, 
                diff_total, diff_politics, diff_english, diff_special_one, diff_special_two, crawl_year
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
            data_type=VALUES(data_type),
            school_name=VALUES(school_name),
            depart_id=VALUES(depart_id),
            depart_name=VALUES(depart_name),
            code=VALUES(code),
            name=VALUES(name),
            politics=VALUES(politics),
            politics_str=VALUES(politics_str),
            english=VALUES(english),
            english_str=VALUES(english_str),
            special_one=VALUES(special_one),
            special_one_str=VALUES(special_one_str),
            special_two=VALUES(special_two),
            special_two_str=VALUES(special_two_str),
            total=VALUES(total),
            note=VALUES(note),
            year=VALUES(year),
            degree_type=VALUES(degree_type),
            special_remark=VALUES(special_remark),
            diff_total=VALUES(diff_total),
            diff_politics=VALUES(diff_politics),
            diff_english=VALUES(diff_english),
            diff_special_one=VALUES(diff_special_one),
            diff_special_two=VALUES(diff_special_two),
            crawl_year=VALUES(crawl_year)
            """
            
            # 同时提取专业信息
            majors_data = set()
            
            # 逐块处理数据
            for i, chunk in enumerate(chunks):
                data_list = []
                
                for _, row in chunk.iterrows():
                    # 提取专业信息
                    if not pd.isna(row['code']) and not pd.isna(row['name']):
                        majors_data.add((row['code'], row['name']))
                    
                    # 处理空值
                    data = (
                        row['id'],
                        row['data_type'],
                        int(row['school_id']),
                        row['school_name'],
                        None if pd.isna(row['depart_id']) else int(row['depart_id']),
                        None if pd.isna(row['depart_name']) else row['depart_name'],
                        None if pd.isna(row['code']) else row['code'],
                        None if pd.isna(row['name']) else row['name'],
                        None if pd.isna(row['politics']) else int(row['politics']),
                        None if pd.isna(row['politics_str']) else row['politics_str'],
                        None if pd.isna(row['english']) else int(row['english']),
                        None if pd.isna(row['english_str']) else row['english_str'],
                        None if pd.isna(row['special_one']) else int(row['special_one']),
                        None if pd.isna(row['special_one_str']) else row['special_one_str'],
                        None if pd.isna(row['special_two']) else int(row['special_two']),
                        None if pd.isna(row['special_two_str']) else row['special_two_str'],
                        None if pd.isna(row['total']) else int(row['total']),
                        None if pd.isna(row['note']) else row['note'],
                        None if pd.isna(row['year']) else int(row['year']),
                        None if pd.isna(row['degree_type']) else row['degree_type'],
                        None if pd.isna(row['special_remark']) else row['special_remark'],
                        None if pd.isna(row['diff_total']) else int(row['diff_total']),
                        None if pd.isna(row['diff_politics']) else int(row['diff_politics']),
                        None if pd.isna(row['diff_english']) else int(row['diff_english']),
                        None if pd.isna(row['diff_special_one']) else int(row['diff_special_one']),
                        None if pd.isna(row['diff_special_two']) else int(row['diff_special_two']),
                        None if pd.isna(row['crawl_year']) else int(row['crawl_year'])
                    )
                    data_list.append(data)
                
                # 批量插入数据
                if data_list:
                    self.cursor.executemany(insert_query, data_list)
                    self.conn.commit()
                    total_count += len(data_list)
                    logger.info(f"已导入 {total_count} 条录取分数数据")
            
            # 插入专业信息
            self._import_majors(majors_data)
            
            logger.info(f"成功导入录取分数数据，共 {total_count} 条记录")
            return total_count
        except Exception as e:
            logger.error(f"导入录取分数数据失败: {e}")
            self.conn.rollback()
            return 0
    
    def _import_majors(self, majors_data):
        """导入专业信息"""
        try:
            if not majors_data:
                return 0
            
            # 准备插入语句
            insert_query = """
            INSERT INTO majors (code, name)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE
            name=VALUES(name)
            """
            
            # 批量插入
            self.cursor.executemany(insert_query, list(majors_data))
            self.conn.commit()
            logger.info(f"成功导入专业信息，共 {len(majors_data)} 条记录")
            return len(majors_data)
        except Exception as e:
            logger.error(f"导入专业信息失败: {e}")
            self.conn.rollback()
            return 0
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("数据库连接已关闭")
    
    def run(self):
        """运行数据库初始化流程"""
        try:
            # 连接数据库
            if not self.connect():
                return False
            
            # 创建数据库
            if not self.create_database():
                return False
            
            # 创建表结构
            if not self.create_tables():
                return False
            
            # 获取CSV文件
            school_file, score_file = self.get_latest_csv_files()
            if not school_file or not score_file:
                return False
            
            # 导入学校数据
            schools_count = self.import_schools_data(school_file)
            
            # 导入录取分数数据
            scores_count = self.import_admission_scores(score_file)
            
            logger.info(f"数据库初始化完成！导入学校 {schools_count} 条，录取分数 {scores_count} 条")
            return True
        except Exception as e:
            logger.error(f"数据库初始化过程出错: {e}")
            return False
        finally:
            self.close()


if __name__ == "__main__":
    print("开始初始化考研分析数据库...")
    initializer = DatabaseInitializer()
    success = initializer.run()
    
    if success:
        print("数据库初始化成功！")
    else:
        print("数据库初始化失败，请查看日志获取详细信息。") 