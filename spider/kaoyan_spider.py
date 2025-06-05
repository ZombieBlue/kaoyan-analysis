# -*- coding: utf-8 -*-

import requests
import pandas as pd
import time
import random
import logging
import logging.handlers
import os
import json
from datetime import datetime
import traceback

# 配置日志
logger = logging.getLogger('KaoyanSpider')
logger.setLevel(logging.INFO)

# 文件日志处理器
file_handler = logging.FileHandler('../logs/kaoyan_spider.log', encoding='utf-8', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加控制台处理器，方便在终端查看
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class KaoyanSpider:
    """考研网站爬虫，获取各学校各专业历年录取分数"""
    
    def __init__(self):
        """初始化爬虫"""
        # 接口URL
        self.school_list_api = "https://api.kaoyan.cn/pc/school/schoolList"
        self.school_score_api = "https://api.kaoyan.cn/pc/school/schoolScore"
        
        # 请求头
        self.headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-TW;q=0.6',
            'content-type': 'application/json;charset=UTF-8',
            'origin': 'https://www.kaoyan.cn',
            'priority': 'u=1, i',
            'referer': 'https://www.kaoyan.cn/',
            'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        }
        
        # 创建数据目录
        self.data_dir = "../data/raw"
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("考研网站爬虫初始化完成")
    
    def get_schools(self, page=1, limit=50):
        """获取学校列表"""
        json_data = {
            'page': page,
            'limit': limit,
            'province_id': '',
            'type': '',
            'feature': '',
            'school_name': '',
        }
        
        try:
            logger.info(f"获取学校列表: 第{page}页，每页{limit}所")
            response = requests.post(self.school_list_api, headers=self.headers, json=json_data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') == '0000':  # 注意API返回的code是字符串'0000'而不是数字
                data = result.get('data', {})
                schools = data.get('data', [])
                total = data.get('total', 0)
                
                logger.info(f"成功获取学校列表: 当前页{len(schools)}所，总计{total}所")
                return schools, total
            else:
                logger.error(f"获取学校列表失败: {result.get('message', '未知错误')}")
                return [], 0
        except Exception as e:
            logger.error(f"获取学校列表异常: {str(e)}")
            return [], 0
    
    def get_school_scores(self, school_id, year):
        """获取学校特定年份的分数数据"""
        json_data = {
            'school_id': str(school_id),  # 转为字符串，确保格式正确
            'year': year,
            'degree_type': '',
        }
        
        try:
            logger.info(f"获取学校(ID: {school_id})在{year}年的分数数据")
            response = requests.post(self.school_score_api, headers=self.headers, json=json_data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') == '0000':
                scores = result.get('data', [])
                logger.info(f"成功获取{year}年分数数据: {len(scores)}条")
                return scores
            else:
                logger.error(f"获取{year}年分数数据失败: {result.get('message', '未知错误')}")
                return []
        except Exception as e:
            logger.error(f"获取{year}年分数数据异常: {str(e)}")
            return []
    
    def crawl_all_schools_scores(self):
        """爬取所有学校的录取分数"""
        # 1. 获取所有学校
        all_schools = []
        page = 1
        limit = 100  # 每页获取100所学校
        total_schools = float('inf')  # 初始值设为无穷大
        
        while len(all_schools) < total_schools:
            # 添加随机延迟，避免请求过于频繁
            time.sleep(random.uniform(1, 3))
            
            schools, total = self.get_schools(page, limit)
            if not schools:
                break
                
            all_schools.extend(schools)
            total_schools = total  # 更新学校总数
            
            logger.info(f"已获取{len(all_schools)}/{total_schools}所学校")
            
            # 如果已经获取了所有学校，就跳出循环
            if len(all_schools) >= total_schools:
                break
                
            page += 1
        
        # 保存学校列表
        if all_schools:
            # 提取需要的字段
            schools_data = [{
                'school_id': school.get('school_id'),
                'school_name': school.get('school_name'),
                'province': school.get('province_name', '未知'),
                'type': school.get('type_name', '未知'),
                'is_985': '是' if school.get('is_985') == 1 else '否',
                'is_211': '是' if school.get('is_211') == 1 else '否',
                'recruit_number': school.get('recruit_number', ''),
                'rank': school.get('rk_rank', '')
            } for school in all_schools]
            
            schools_df = pd.DataFrame(schools_data)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            schools_file = os.path.join(self.data_dir, f"school_list_{timestamp}.csv")
            schools_df.to_csv(schools_file, index=False, encoding='utf-8-sig')
            logger.info(f"学校列表已保存: {schools_file}")
        
        # 2. 获取每所学校近三年的分数 (2022-2024)
        years = [2022, 2023, 2024]
        all_scores = []
        
        for i, school in enumerate(all_schools):
            school_id = school.get('school_id')
            school_name = school.get('school_name')
            
            if not school_id or not school_name:
                continue
            
            logger.info(f"处理学校 [{i+1}/{len(all_schools)}]: {school_name}")
            
            school_scores = []
            for year in years:
                # 添加随机延迟，避免请求过于频繁
                time.sleep(random.uniform(1, 3))
                
                scores = self.get_school_scores(school_id, year)
                if scores:
                    # 添加年份标记，确保数据完整性
                    for score in scores:
                        score['crawl_year'] = year
                    
                    school_scores.extend(scores)
            
            if school_scores:
                all_scores.extend(school_scores)
                logger.info(f"已获取{school_name}历年分数记录{len(school_scores)}条")
            
            # 每处理10所学校保存一次数据，防止数据丢失
            if (i + 1) % 10 == 0 and all_scores:
                self._save_temp_scores(all_scores, i + 1)
        
        # 3. 保存所有分数数据
        if all_scores:
            final_file = self._save_final_scores(all_scores)
            logger.info(f"所有分数数据已保存: {final_file}")
        
        return len(all_schools), len(all_scores)
    
    def _save_temp_scores(self, scores, processed_count):
        """保存临时分数数据"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_file = os.path.join(self.data_dir, f"admission_scores_temp_{processed_count}_{timestamp}.csv")
        
        scores_df = pd.DataFrame(scores)
        # 使用utf-8-sig编码保存CSV文件，确保Excel正确识别中文
        scores_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"临时分数数据已保存: {temp_file}, 包含{len(scores)}条记录")
    
    def _save_final_scores(self, scores):
        """保存最终分数数据"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        final_file = os.path.join(self.data_dir, f"admission_scores_full_{timestamp}.csv")
        
        scores_df = pd.DataFrame(scores)
        # 使用utf-8-sig编码保存CSV文件，确保Excel正确识别中文
        scores_df.to_csv(final_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"最终分数数据已保存: {final_file}, 共{len(scores)}条记录")
        return final_file
    
    def run(self):
        """运行爬虫主函数"""
        logger.info("开始运行考研网站分数爬虫...")
        start_time = datetime.now()
        
        try:
            schools_count, scores_count = self.crawl_all_schools_scores()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60  # 计算运行时间（分钟）
            
            logger.info(f"爬虫运行完成! 共获取{schools_count}所学校，{scores_count}条分数记录")
            logger.info(f"总耗时: {duration:.2f}分钟")
            
            return True
        except Exception as e:
            logger.error(f"爬虫运行出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    spider = KaoyanSpider()
    spider.run()