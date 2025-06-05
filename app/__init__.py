#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask_cors import CORS

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 启用跨域资源共享
    
    # 注册API蓝图
    from app.api import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    @app.route('/')
    def index():
        return "考研数据分析系统API"
    
    return app 