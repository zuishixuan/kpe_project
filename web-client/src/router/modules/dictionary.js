/** When your routing table is too long, you can split it into small modules **/

import Layout from '@/layout'

const dictRouter =
  {
    path: '/dictionary',
    component: Layout,
    redirect: '/dictionary/keyword_dict',
    name: 'dictionary',
    meta: { roles: ['admin'], title: '词库管理', icon: 'el-icon-s-help' },
    children: [
      {
        path: 'keyword_dict',
        name: 'keyword_dict',
        component: () => import('@/views/dictionary/keyword_dict/index'),
        meta: { title: '关键词库', icon: 'table' }
      },
      {
        path: 'synonym_dict',
        name: 'synonym_dict',
        component: () => import('@/views/dictionary/synonym_dict/index'),
        meta: { title: '同义词库', icon: 'table' }
      },
      {
        path: 'hotword_dict',
        name: 'hotword_dict',
        component: () => import('@/views/dictionary/hotword_dict/index'),
        meta: { title: '热词库', icon: 'table' }
      }
    ]
  }

export default dictRouter
