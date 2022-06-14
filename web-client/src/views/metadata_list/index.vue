<template>
  <el-card class="box-card" shadow="hover">
    <div style="margin-top: 15px;">
      <el-input v-model="search.search_word" placeholder="请输入内容" class="input-with-select" style="width: 600px;margin-left: 400px">
        <el-select slot="prepend" v-model="search.search_select" placeholder="请选择" style="width: 120px">
          <el-option label="全部" value="0" />
          <el-option label="标题" value="name" />
          <el-option label="摘要描述" value="describe" />
          <el-option label="关键词" value="keyword" />
        </el-select>
        <el-button slot="append" icon="el-icon-search" @click="doSearch" />
      </el-input>
    </div>
    <el-table
      :data="tableData"
      border
      height="650"
      style="width: 100%;margin-top: 10px"
    >
      <el-table-column
        align="center"
        prop="identifier"
        label="标识符"
        width="200"
      />
      <el-table-column
        align="center"
        fixed
        prop="name"
        label="标题"
        width="200"
      />
      <el-table-column
        align="center"
        prop="subject_category"
        label="学科分类"
        width="150"
      />
      <el-table-column
        align="center"
        prop="theme_category"
        label="主题分类"
        width="200"
      />
      <el-table-column
        align="center"
        prop="describe"
        label="摘要描述"
        width="300"
      />
      <el-table-column
        align="center"
        prop="keyword"
        label="关键词"
        width="200"
        filter-placement="bottom-end"
      >
        <template slot-scope="scope">
          <el-tag
            v-for="item in scope.row.keyword"
            :type="'success'"
            disable-transitions
            size="mini"
          >{{ item }}</el-tag>
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
import { getEsMatchPage, getEsPage } from '@/api/metadata'

export default {

  data() {
    return {
      page: 1,
      offset: 10,
      totalNum: 1000,
      tableData: [],
      search_word: '',
      search: {
        search_word: '',
        search_select: '0'
      }
    }
  },
  created() {
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
      getEsPage(params).then(response => {
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
      console.log(index, row)
    },
    handleDelete(index, row) {
      console.log(index, row)
    },
    doSearch() {
      const params = {
        pageno: this.page,
        pagesize: this.offset,
        search_word: this.search.search_word,
        search_select: this.search.search_select,
        username: this.$store.getters.name
      }
      console.log(params)
      getEsMatchPage(params).then(response => {
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
    }

  }
}
</script>

<style scoped>
  .el-select .el-input {
    width: 330px;
  }
  .input-with-select .el-input-group__prepend {
    background-color: #fff;
  }
</style>
