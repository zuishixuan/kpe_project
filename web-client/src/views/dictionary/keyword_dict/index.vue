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
        <el-form-item label="权重">
          <el-input v-model="form.weight" placeholder="请输入权重" />
        </el-form-item>
        <el-form-item label="状态" prop="resource">
          <el-radio-group v-model="form.status">
            <el-radio label="0">锁定启用</el-radio>
            <el-radio label="1">正常启用</el-radio>
            <el-radio label="2">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="createDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="createKeyword">确 定</el-button>
      </span>
    </el-dialog>

    <el-dialog
      title="编辑关键词"
      :visible.sync="updateDialogVisible"
      width="30%"
      :before-close="handleClose"
    >
      <el-form ref="form2" :model="form2" label-width="100px" style="margin-top: 30px">
        <el-form-item label="关键词">
          <el-input v-model="form2.word" placeholder="请输入关键词" />
        </el-form-item>
        <el-form-item label="权重">
          <el-input v-model="form2.weight" placeholder="请输入权重" />
        </el-form-item>
        <el-form-item label="状态" prop="resource">
          <el-radio-group v-model="form2.status">
            <el-radio label="0">锁定启用</el-radio>
            <el-radio label="1">正常启用</el-radio>
            <el-radio label="2">禁用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="updateDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="updateKeyword">确 定</el-button>
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
        align="center"
        show-password
        prop="weight"
        label="权重"
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
      <el-table-column align="center" label="操作">
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
import {getPage, saveKeyword, deleteKeyword, updateKeyword} from '@/api/keyword-dict'

export default {

  data() {
    return {
      page: 1,
      offset: 10,
      totalNum: 1000,
      tableData: [],
      showTag: ['锁定启用', '正常启用', '禁用'],
      showStyle: ['success', 'info', 'danger'],
      createDialogVisible: false,
      updateDialogVisible: false,
      form: {
        word: '',
        weight: '',
        status: 2,
        add_date: ''
      },
      form2: {
        word: '',
        weight: '',
        status: 2,
        add_date: ''
      }
    }
  },
  created() {
    this.clearForm()
    this.filter()
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
      deleteKeyword(row.id).then(response => {
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
    filterTag(value, row) {
      return row.tag === value
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
        weight: '',
        status: '2',
        add_date: (new Date()).toLocaleDateString()
      }
    },
    createKeyword() {
      this.createDialogVisible = false
      saveKeyword(this.form).then(response => {
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
    updateKeyword() {
      this.updateDialogVisible = false
      updateKeyword(this.form2).then(response => {
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
