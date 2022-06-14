<template>
  <el-card class="box-card" shadow="hover">
    <el-button type="primary" @click="createDialogVisible = true">添加关键词</el-button>

    <el-dialog
      title="添加关键词"
      :visible.sync="createDialogVisible"
      width="30%"
      :before-close="handleClose"
    >
      <el-form ref="form" :model="form" label-width="100px" style="margin-top: 30px">
        <el-form-item label="关键词">
          <el-input v-model="form.word" placeholder="请输入关键词" />
        </el-form-item>
        <el-form-item label="主要关键词">
          <el-input v-model="form.main_word" placeholder="请输入主要关键词" />
        </el-form-item>
        <el-form-item label="状态" prop="resource">
          <el-radio-group v-model="form.status">
            <el-radio label="0">启用</el-radio>
            <el-radio label="1">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="createDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="createSynonym">确 定</el-button>
      </span>
    </el-dialog>

    <el-dialog
      title="添加关键词"
      :visible.sync="updateDialogVisible"
      width="30%"
      :before-close="handleClose"
    >
      <el-form ref="form" :model="form2" label-width="100px" style="margin-top: 30px">
        <el-form-item label="关键词">
          <el-input v-model="form2.word" placeholder="请输入关键词" />
        </el-form-item>
        <el-form-item label="主要关键词">
          <el-input v-model="form2.main_word" placeholder="请输入主要关键词" />
        </el-form-item>
        <el-form-item label="状态" prop="resource">
          <el-radio-group v-model="form2.status">
            <el-radio label="0">启用</el-radio>
            <el-radio label="1">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="updateDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="updateSynonym">确 定</el-button>
      </span>
    </el-dialog>

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
        width="300"
      />
      <el-table-column
        align="center"
        prop="word"
        label="关键词"
        width="300"
      />
      <el-table-column
        show-password
        align="center"
        prop="main_word"
        label="主要关键词"
        width="350"
      />
      <el-table-column
        align="center"
        prop="status"
        label="状态"
        width="300"
        filter-placement="bottom-end"
      >
        <template slot-scope="scope">
          <el-tag
            :type="showStyle[scope.row.status]"
            disable-transitions
          >{{ showTag[scope.row.status] }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column a lign="center" label="操作">
        <template slot-scope="scope">
          <el-button
            size="mini"
            @click="handleEdit(scope.$index, scope.row)"
          >编辑</el-button>
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
import { getPage, saveSynonym, deleteSynonym, updateSynonym } from '@/api/synonym-dict'

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
      deleteSynonym(row.id).then(response => {
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
      saveSynonym(this.form).then(response => {
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
      updateSynonym(this.form2).then(response => {
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
