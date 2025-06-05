import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: Dashboard
  },
  {
    path: '/schools',
    name: 'schools',
    component: () => import('../views/SchoolAnalysis.vue')
  },
  {
    path: '/majors',
    name: 'majors',
    component: () => import('../views/MajorAnalysis.vue')
  },
  {
    path: '/provinces',
    name: 'provinces',
    component: () => import('../views/ProvinceAnalysis.vue')
  },
  {
    path: '/scores',
    name: 'scores',
    component: () => import('../views/ScoreAnalysis.vue')
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router 