<template>
  <div class="chart-card" :style="cardStyle">
    <div class="chart-header">
      <div class="chart-title">
        <h3>{{ title }}</h3>
      </div>
      <div class="chart-tools" v-if="showTools">
        <el-tooltip content="刷新数据">
          <el-button circle size="small" @click="$emit('refresh')">
            <el-icon><Refresh /></el-icon>
          </el-button>
        </el-tooltip>
        <el-tooltip content="全屏查看">
          <el-button circle size="small" @click="toggleFullScreen">
            <el-icon><FullScreen /></el-icon>
          </el-button>
        </el-tooltip>
      </div>
    </div>
    <div class="chart-content">
      <slot></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChartCard',
  props: {
    title: {
      type: String,
      required: true
    },
    showTools: {
      type: Boolean,
      default: true
    },
    height: {
      type: String,
      default: '400px'
    }
  },
  computed: {
    cardStyle() {
      return {
        height: this.height
      }
    }
  },
  methods: {
    toggleFullScreen() {
      // 实现全屏功能
      const element = this.$el
      
      if (!document.fullscreenElement) {
        if (element.requestFullscreen) {
          element.requestFullscreen()
        } else if (element.webkitRequestFullscreen) {
          element.webkitRequestFullscreen()
        } else if (element.msRequestFullscreen) {
          element.msRequestFullscreen()
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen()
        } else if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen()
        } else if (document.msExitFullscreen) {
          document.msExitFullscreen()
        }
      }
    }
  }
}
</script>

<style scoped>
.chart-card {
  background-color: #fff;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid #f0f0f0;
}
.chart-title h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 500;
  color: #1e1e1e;
}
.chart-content {
  flex: 1;
  padding: 20px;
  overflow: hidden;
}
</style> 