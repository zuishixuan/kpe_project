<template>
  <el-card class="box-card" shadow="hover">

    <el-table
      :data="tableData"
      border
      height="650"
      style="width: 100%;margin-top: 10px"
    >
      <el-table-column
        fixed
        align="center"
        prop="id"
        label="id"
        width="100"
      />
      <el-table-column
        align="center"
        prop="kpe_record_id"
        label="关键词抽取记录编号"
        width="200"
      />
      <el-table-column
        align="center"
        prop="name"
        label="名称"
        width="350"
      />
      <el-table-column
        align="center"
        prop="keyword"
        label="反馈关键词"
        width="350">
        <template slot-scope="scope">
          <el-tag
            :type="'success'"
            disable-transitions
          >{{scope.row.keyword}}</el-tag>
        </template>
      </el-table-column>
      <el-table-column
        align="center"
        prop="score"
        label="反馈分数"
        width="250"
      />
      <el-table-column
        align="center"
        prop="submit_date"
        label="日期"
        width="300"
      />
      <el-table-column align="center" label="操作">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="danger"
            @click="handleDelete(scope.$index, scope.row)"
          >删除</el-button>
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
import { getPage, saveFeedbackRecord, deleteFeedbackRecord, updateFeedbackRecord } from '@/api/feedback-record'

export default {

  data() {
    return {
      page: 1,
      offset: 10,
      totalNum: 1000,
      tableData: [],
      showTag: ['启用', '禁用'],
      showStyle: ['success', 'danger'],
      createDialogVisible: false,
      updateDialogVisible: false,
      form: {
        word: '',
        main_word: '',
        status: 1,
        add_date: ''
      },
      form2: {
        word: '',
        main_word: '',
        status: 1,
        add_date: ''
      }
    }
  },
  created() {
    this.filter()
    this.clearForm()
  },
  methods: {
    handleClick(row) {
      console.log(row)
    },
    filter() {
      const params = {
        pageno: this.page,
        pagesize: this.offset
      }
      getPage(params).then(response => {
        const h = this.$createElement
        this.tableData = response.data.data
        console.log(this.tableData)
        this.totalNum = response.data['totalnum']
        this.$notify({
          title: '搜索成功',
          message: h('i', { style: 'color: teal' }, '搜索成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
    },
    handleSizeChange(val) {
      this.offset = val
      this.filter()
    },
    handleCurrentChange(val) {
      this.page = val
      this.filter()
    },
    handleEdit(index, row) {
      this.form2 = JSON.parse(JSON.stringify(row))
      this.updateDialogVisible = true
    },
    handleDelete(index, row) {
      deleteFeedbackRecord(row.id).then(response => {
        const h = this.$createElement
        this.ex_flag = true
        this.filter()
        this.$notify({
          title: '删除成功',
          message: h('i', { style: 'color: teal' }, '删除成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
    },
    handleClose(done) {
      this.$confirm('确认关闭？')
        .then(_ => {
          done()
        })
        .catch(_ => {
        })
    },
    clearForm() {
      this.form = {
        word: '',
        main_word: '',
        status: 1,
        add_date: (new Date()).toLocaleDateString()
      }
    },
    createSynonym() {
      this.createDialogVisible = false
      saveFeedbackRecord(this.form).then(response => {
        const h = this.$createElement
        this.ex_flag = true
        this.clearForm()
        this.filter()
        this.$notify({
          title: '添加成功',
          message: h('i', { style: 'color: teal' }, '添加成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
    },
    updateSynonym() {
      this.updateDialogVisible = false
      updateFeedbackRecord(this.form2).then(response => {
        const h = this.$createElement
        this.ex_flag = true
        this.filter()
        this.$notify({
          title: '修改成功',
          message: h('i', { style: 'color: teal' }, '修改成功'),
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
