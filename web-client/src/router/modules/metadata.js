import Layout from '@/layout'

const metadataRouter =
  {
    path: '/metadata',
    component: Layout,
    children: [
      {
        path: 'index',
        name: 'metadata',
        component: () => import('@/views/metadata/index'),
        meta: { roles: ['admin', 'editor'], title: '科技资源提交', icon: 'form' }
      }
    ]
  }

export default metadataRouter
