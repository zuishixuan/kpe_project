<template>
  <el-card class="box-card" shadow="hover">
    <el-button type="primary" @click="createDialogVisible = true">添加新用户</el-button>

    <el-dialog
      title="创建用户"
      :visible.sync="createDialogVisible"
      width="30%"
      :before-close="handleClose"
    >
      <el-form ref="form" :model="form" label-width="100px" style="margin-top: 30px">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.pwd" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-form-item label="权限" prop="resource">
          <el-radio-group v-model="form.role">
            <el-radio label="admin" />
            <el-radio label="editor" />
            <el-radio label="visitor" />
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="handleClose = false">取 消</el-button>
        <el-button type="primary" @click="createUser">确 定</el-button>
      </span>
    </el-dialog>

    <el-dialog
      title="编辑用户"
      :visible.sync="updateDialogVisible"
      width="30%"
      :before-close="handleClose"
    >
      <el-form ref="form" :model="form2" label-width="100px" style="margin-top: 30px">
        <el-form-item label="用户名">
          <el-input v-model="form2.username" placeholder="请输入用户名" disabled />
        </el-form-item>
        <el-form-item label="权限" prop="resource">
          <el-radio-group v-model="form2.role">
            <el-radio label="admin" />
            <el-radio label="editor" />
            <el-radio label="visitor" />
          </el-radio-group>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="handleClose = false">取 消</el-button>
        <el-button type="primary" @click="updateUser">确 定</el-button>
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
        width="200"
      />
      <el-table-column
        align="center"
        prop="username"
        label="用户名"
        width="400"
      />
      <el-table-column
        align="center"
        prop="role"
        label="权限类别"
        width="400"
      />
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
import { getUserPage, saveUser, deleteUser, updateUser } from '@/api/user'

export default {
  data() {
    return {
      page: 1,
      offset: 10,
      totalNum: 1000,
      tableData: [],
      createDialogVisible: false,
      updateDialogVisible: false,
      form: {
        username: '',
        pwd: '',
        role: 'visitor',
        add_date: ''
      },
      form2: {
        username: '',
        pwd: '',
        role: '',
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
      getUserPage(params).then(response => {
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
      deleteUser(row.id).then(response => {
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
    clearForm() {
      this.form = {
        username: '',
        pwd: '',
        role: 'visitor',
        add_date: (new Date()).toLocaleDateString()
      }
    },
    handleClose(done) {
      this.$confirm('确认关闭？')
        .then(_ => {
          done()
        })
        .catch(_ => {
        })
    },
    createUser() {
      this.createDialogVisible = false
      saveUser(this.form).then(response => {
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
    updateUser() {
      this.updateDialogVisible = false
      updateUser(this.form2).then(response => {
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
