import Layout from '@/layout'

const metadataListRouter =
  {
    path: '/metadata_list',
    component: Layout,
    children: [
      {
        path: 'index',
        name: 'metadata_list',
        component: () => import('@/views/metadata_list/index'),
        meta: { title: '科技资源目录', icon: 'form' }
      }
    ]
  }

export default metadataListRouter
