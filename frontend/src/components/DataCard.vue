<template>
  <div class="data-card" :style="{ backgroundColor: bgColor }">
    <div class="card-icon">
      <el-icon :size="36" :color="iconColor">
        <component :is="icon"></component>
      </el-icon>
    </div>
    <div class="card-content">
      <div class="card-title">{{ title }}</div>
      <div class="card-value">{{ value }}</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DataCard',
  props: {
    title: {
      type: String,
      required: true
    },
    value: {
      type: [Number, String],
      required: true
    },
    icon: {
      type: String,
      required: true
    },
    bgColor: {
      type: String,
      default: '#67C23A'
    },
    iconColor: {
      type: String,
      default: '#ffffff'
    }
  }
}
</script>

<style scoped>
.data-card {
  display: flex;
  flex-direction: row;
  align-items: center;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  padding: 20px;
  color: #ffffff;
  height: 120px;
}
.card-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.2);
  margin-right: 20px;
}
.card-content {
  display: flex;
  flex-direction: column;
}
.card-title {
  font-size: 16px;
  margin-bottom: 8px;
}
.card-value {
  font-size: 28px;
  font-weight: bold;
}
</style> 