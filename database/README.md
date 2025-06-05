# 考研分析数据库模块

此模块用于初始化MySQL数据库，创建表结构并导入CSV数据。

## 功能

- 自动创建数据库和表结构
- 导入学校信息数据
- 导入历年考研录取分数数据
- 提取专业信息

## 表结构

### schools 表（学校信息）

| 字段名 | 类型 | 说明 |
|-------|------|------|
| school_id | INT | 学校ID（主键） |
| school_name | VARCHAR(100) | 学校名称 |
| province | VARCHAR(50) | 所在省份 |
| type | VARCHAR(50) | 学校类型 |
| is_985 | ENUM('是', '否') | 是否985院校 |
| is_211 | ENUM('是', '否') | 是否211院校 |
| recruit_number | INT | 招生人数 |
| ranking | INT | 排名 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### admission_scores 表（录取分数）

| 字段名 | 类型 | 说明 |
|-------|------|------|
| id | VARCHAR(20) | 记录ID（主键） |
| data_type | VARCHAR(20) | 数据类型 |
| school_id | INT | 学校ID（外键） |
| school_name | VARCHAR(100) | 学校名称 |
| depart_id | INT | 院系ID |
| depart_name | VARCHAR(100) | 院系名称 |
| code | VARCHAR(20) | 专业代码 |
| name | VARCHAR(100) | 专业名称 |
| politics | INT | 政治分数 |
| politics_str | VARCHAR(20) | 政治分数描述 |
| english | INT | 英语分数 |
| english_str | VARCHAR(20) | 英语分数描述 |
| special_one | INT | 专业课一分数 |
| special_one_str | VARCHAR(20) | 专业课一分数描述 |
| special_two | INT | 专业课二分数 |
| special_two_str | VARCHAR(20) | 专业课二分数描述 |
| total | INT | 总分 |
| note | TEXT | 备注 |
| year | INT | 年份 |
| degree_type | VARCHAR(10) | 学位类型 |
| special_remark | TEXT | 特殊备注 |
| diff_total | INT | 总分差异 |
| diff_politics | INT | 政治分数差异 |
| diff_english | INT | 英语分数差异 |
| diff_special_one | INT | 专业课一分数差异 |
| diff_special_two | INT | 专业课二分数差异 |
| crawl_year | INT | 爬取年份 |
| created_at | TIMESTAMP | 创建时间 |

### majors 表（专业信息）

| 字段名 | 类型 | 说明 |
|-------|------|------|
| code | VARCHAR(20) | 专业代码（主键） |
| name | VARCHAR(100) | 专业名称 |
| category | VARCHAR(50) | 专业类别 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置数据库连接：
编辑 `../config/database.ini` 文件，设置正确的数据库连接信息。

3. 运行初始化脚本：
```bash
python db_init.py
```

## 注意事项

- 确保MySQL服务已启动
- 确保用户有创建数据库的权限
- CSV数据文件应位于 `../data/raw` 目录下 