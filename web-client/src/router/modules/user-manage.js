import Layout from '@/layout'

const userManageRouter =
  {
    path: '/user-manage',
    component: Layout,
    children: [
      {
        path: 'index',
        name: 'user-manage',
        component: () => import('@/views/user-manage/index'),
        meta: { roles: ['admin'], title: '用户管理', icon: 'form' }
      }
    ]
  }

export default userManageRouter
