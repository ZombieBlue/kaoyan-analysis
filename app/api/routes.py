#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from app.api import api_blueprint
from app.api.data_service import KaoyanDataService

# 创建数据服务实例
data_service = KaoyanDataService()

@api_blueprint.route('/top-schools', methods=['GET'])
def get_top_schools():
    """获取热门学校Top20"""
    top_schools = data_service.get_top_schools(limit=20)
    return jsonify({
        'code': 200,
        'data': top_schools,
        'message': '获取热门学校排名成功'
    })

@api_blueprint.route('/top-majors', methods=['GET'])
def get_top_majors():
    """获取热门专业Top20"""
    top_majors = data_service.get_top_majors(limit=20)
    return jsonify({
        'code': 200,
        'data': top_majors,
        'message': '获取热门专业排名成功'
    })

@api_blueprint.route('/top-provinces', methods=['GET'])
def get_top_provinces():
    """获取热门省份Top20"""
    top_provinces = data_service.get_top_provinces(limit=20)
    return jsonify({
        'code': 200,
        'data': top_provinces,
        'message': '获取热门省份排名成功'
    })

@api_blueprint.route('/all-stats', methods=['GET'])
def get_all_stats():
    """获取所有统计数据"""
    all_stats = data_service.get_all_stats()
    return jsonify({
        'code': 200,
        'data': all_stats,
        'message': '获取所有统计数据成功'
    }) 