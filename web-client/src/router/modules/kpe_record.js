/** When your routing table is too long, you can split it into small modules **/

import Layout from '@/layout'

const kpeRecordRouter =
  {
    path: '/kpe_feedback_record',
    component: Layout,
    redirect: '/kpe_feedback_record/kpe_record',
    name: 'dictionary',
    meta: { roles: ['admin'], title: '关键词抽取记录与反馈', icon: 'el-icon-s-help' },
    children: [
      {
        path: 'kpe_record',
        name: 'kpe_record',
        component: () => import('@/views/kpe_feedback_record/kpe_record/index'),
        meta: { title: '关键词抽取记录', icon: 'table' }
      },
      {
        path: 'feedback_record',
        name: 'feedback_record',
        component: () => import('@/views/kpe_feedback_record/feedback_record/index'),
        meta: { title: '关键词抽取反馈', icon: 'table' }
      }
    ]
  }

export default kpeRecordRouter
