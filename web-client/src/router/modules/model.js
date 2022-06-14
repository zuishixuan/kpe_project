import Layout from '@/layout'

const modelRouter =
  {
    path: '/model',
    component: Layout,
    redirect: '/model/parameter',
    name: 'model',
    meta: { roles: ['admin'], title: '模型设置', icon: 'el-icon-s-help' },
    children: [
      {
        path: 'parameter',
        name: 'parameter',
        component: () => import('@/views/model/parameter/index'),
        meta: { title: '模型参数调节', icon: 'table' }
      }
    ]
  }

export default modelRouter
