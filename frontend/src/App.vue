<template>
  <div class="app-container">
    <el-container class="container">
      <el-aside width="200px" class="sidebar">
        <div class="logo">
          <h2>考研数据分析</h2>
        </div>
        <el-menu
          router
          :default-active="activeMenu"
          class="menu"
          background-color="#001529"
          text-color="#fff"
          active-text-color="#ffd04b">
          <el-menu-item index="/">
            <el-icon><DataLine /></el-icon>
            <span>数据总览</span>
          </el-menu-item>
          <el-menu-item index="/schools">
            <el-icon><School /></el-icon>
            <span>热门学校</span>
          </el-menu-item>
          <el-menu-item index="/majors">
            <el-icon><Reading /></el-icon>
            <span>热门专业</span>
          </el-menu-item>
          <el-menu-item index="/provinces">
            <el-icon><Location /></el-icon>
            <span>热门省份</span>
          </el-menu-item>
          <el-menu-item index="/scores">
            <el-icon><Histogram /></el-icon>
            <span>分数预测</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      
      <el-container>
        <el-header class="header">
          <div class="header-title">
            <h1>考研数据可视化分析平台</h1>
          </div>
          <div class="header-info">
            <el-dropdown>
              <span class="user-info">
                <el-avatar icon="el-icon-user-solid" size="small"></el-avatar>
                <span class="username">管理员</span>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item>个人信息</el-dropdown-item>
                  <el-dropdown-item>退出系统</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </el-header>
        
        <el-main class="main">
          <router-view />
        </el-main>
        
        <el-footer class="footer">
          Copyright © 2023 考研数据分析系统
        </el-footer>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import { DataLine, School, Reading, Location, Histogram } from '@element-plus/icons-vue'

export default {
  name: 'App',
  components: {
    DataLine,
    School,
    Reading,
    Location,
    Histogram
  },
  computed: {
    activeMenu() {
      return this.$route.path
    }
  }
}
</script>

<style scoped>
.app-container {
  height: 100vh;
  width: 100vw;
}
.container {
  height: 100%;
}
.sidebar {
  background-color: #001529;
  color: white;
  box-shadow: 2px 0 6px rgba(0, 21, 41, 0.35);
}
.logo {
  height: 60px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #ffd04b;
}
.menu {
  border-right: none;
}
.header {
  background-color: #fff;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}
.header-title h1 {
  font-size: 18px;
  color: #1e1e1e;
}
.user-info {
  display: flex;
  align-items: center;
  cursor: pointer;
}
.username {
  margin-left: 10px;
}
.main {
  background-color: #f0f2f5;
  padding: 20px;
  overflow: auto;
}
.footer {
  text-align: center;
  line-height: 60px;
  color: #666;
  font-size: 14px;
  background-color: #f7f7f7;
}
</style> 