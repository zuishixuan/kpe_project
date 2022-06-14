<template>
  <el-card class="box-card" shadow="hover">
    <div class="block" style="display:inline-block">
      <el-date-picker
        v-model="value"
        type="daterange"
        align="right"
        unlink-panels
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        :picker-options="pickerOptions"
      />
    </div>
    <el-button type="primary" style="margin-left: 10px" @click="doSearch">查找热词</el-button>

    <el-table
      :data="tableData"
      border
      height="650"
      style="width: 100%;margin-top: 10px"
    >
      <el-table-column
        align="center"
        prop="word"
        label="关键词"
        width="500"
      />
      <el-table-column
        align="center"
        prop="frequency"
        label="频率"
        width="500"
      />
      <el-table-column label="操作" align="center">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="success"
            @click="handleEdit(scope.$index, scope.row)"
          >加入关键词库</el-button>
        </template>
      </el-table-column>
    </el-table>
    <div class="block" style="float: right">
      <el-pagination
        background
        :current-page="this.page"
        :page-sizes="[10, 50, 100, 200]"
        :page-size="this.offset"
        layout="total, sizes, prev, pager, next, jumper"
        :total="this.totalNum"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>
  </el-card>
</template>
<script>
import { getPage } from '@/api/hotword_dict'
import { saveKeyword } from '@/api/keyword-dict'
import { formatDateTime } from '@/utils/data'

export default {
  data() {
    return {
      tableData: [],
      pickerOptions: {
        shortcuts: [{
          text: '最近一周',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 7)
            picker.$emit('pick', [start, end])
          }
        }, {
          text: '最近一个月',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 30)
            picker.$emit('pick', [start, end])
          }
        }, {
          text: '最近三个月',
          onClick(picker) {
            const end = new Date()
            const start = new Date()
            start.setTime(start.getTime() - 3600 * 1000 * 24 * 90)
            picker.$emit('pick', [start, end])
          }
        }]
      },
      value: ''
    }
  },
  methods: {
    handleEdit(index, row) {
      console.log(index, row)
      const params = {
        'word': row.word,
        'weight': 5,
        'status': 1

      }
      saveKeyword(params).then(response => {
        const h = this.$createElement
        this.ex_flag = true
        this.$notify({
          title: '添加成功',
          message: h('i', { style: 'color: teal' }, '添加成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
    },
    handleDelete(index, row) {
      console.log(index, row)
    },
    doSearch() {
      console.log(this.value)
      const be = formatDateTime(new Date(this.value[0]))
      const end = formatDateTime(new Date(this.value[1]))
      const params = {
        be: be,
        end: end
      }
      getPage(params).then(response => {
        const h = this.$createElement
        this.tableData = response.data
        console.log(this.tableData)
        this.$notify({
          title: '搜索成功',
          message: h('i', { style: 'color: teal' }, '搜索成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
    }
  }
}
</script>

<style scoped>

</style>
