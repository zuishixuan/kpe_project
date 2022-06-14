<template>
  <el-card class="box-card" shadow="hover">
    <!--    <el-button type="primary" @click="createDialogVisible = true">添加关键词</el-button>-->

    <!--    <el-dialog-->
    <!--      title="添加关键词"-->
    <!--      :visible.sync="createDialogVisible"-->
    <!--      width="30%"-->
    <!--      :before-close="handleClose"-->
    <!--    >-->
    <!--      <el-form ref="form" :model="form" label-width="100px" style="margin-top: 30px">-->
    <!--        <el-form-item label="关键词">-->
    <!--          <el-input v-model="form.word" placeholder="请输入关键词" />-->
    <!--        </el-form-item>-->
    <!--        <el-form-item label="权重">-->
    <!--          <el-input v-model="form.weight" placeholder="请输入权重" />-->
    <!--        </el-form-item>-->
    <!--        <el-form-item label="状态" prop="resource">-->
    <!--          <el-radio-group v-model="form.status">-->
    <!--            <el-radio label="0">锁定启用</el-radio>-->
    <!--            <el-radio label="1">正常启用</el-radio>-->
    <!--            <el-radio label="2">禁用</el-radio>-->
    <!--          </el-radio-group>-->
    <!--        </el-form-item>-->
    <!--      </el-form>-->
    <!--      <span slot="footer" class="dialog-footer">-->
    <!--        <el-button @click="createDialogVisible = false">取 消</el-button>-->
    <!--        <el-button type="primary" @click="createKeyword">确 定</el-button>-->
    <!--      </span>-->
    <!--    </el-dialog>-->

    <el-dialog
      title="查看反馈"
      :visible.sync="updateDialogVisible"
      width="45%"
      :before-close="handleClose"
    >
      <el-table
        :data="form2"
        border
        height="600"
        style="width: 100%;margin-top: 10px"
      >
        <el-table-column
          fixed
          align="center"
          prop="id"
          label="id"
          width="100"
        />
        <!--        <el-table-column-->
        <!--          align="center"-->
        <!--          prop="kpe_record_id"-->
        <!--          label="关键词抽取记录编号"-->
        <!--          width="300"-->
        <!--        />-->
        <el-table-column
          align="center"
          prop="name"
          label="名称"
          width="250"
        />
        <el-table-column
          align="center"
          prop="keyword"
          label="反馈关键词"
          width="250"
        >
          <template slot-scope="scope">
            <el-tag
              :type="'success'"
              disable-transitions
            >{{ scope.row.keyword }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column
          align="center"
          prop="score"
          label="反馈分数"
          width="150"
        />
        <el-table-column
          align="center"
          prop="submit_date"
          label="日期"
          width="150"
        />
      </el-table>
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
        width="100"
      />
      <el-table-column
        align="center"
        prop="name"
        label="名称"
        width="300"
      />
      <el-table-column
        align="center"
        prop="describe"
        label="摘要"
        width="400"
      />
      <el-table-column
        align="center"
        prop="keyword"
        label="关键词"
        width="250"
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
      <el-table-column
        align="center"
        prop="keyword_kpe"
        label="抽取关键词"
        width="250"
        filter-placement="bottom-end"
      >
        <template slot-scope="scope">
          <el-tag
            v-for="item in scope.row.keyword_kpe"
            :type="'success'"
            disable-transitions
            size="mini"
          >{{ item }} </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        align="center"
        prop="keyword_score"
        label="评价"
        width="150"
        filter-placement="bottom-end"
      >
        <template slot-scope="scope">
          <el-tag
            :type="showStyle[getScoreLevel(scope.row.keyword_score)]"
            disable-transitions
          >{{ showTag[getScoreLevel(scope.row.keyword_score)] }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column
        align="center"
        prop="submit_date"
        label="日期"
        width="150"
      />
      <el-table-column align="center" label="操作">
        <template slot-scope="scope">
          <el-button
            size="mini"
            @click="handleEdit(scope.$index, scope.row)"
          >查看反馈</el-button>
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
import { getPage, deleteKpeRecord } from '@/api/kpe-record'
import { getFeedbackRecordByKpeRecordId } from '@/api/feedback-record'

export default {

  data() {
    return {
      page: 1,
      offset: 10,
      totalNum: 1000,
      tableData: [],
      showTag: ['优秀', '合格', '不合格'],
      showStyle: ['success', 'info', 'danger'],
      scoreThreshold: [0.75, 0.6],
      createDialogVisible: false,
      updateDialogVisible: false,
      form: {
      },
      form2: {
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
    getScoreLevel(score) {
      console.log(score)
      if (score >= this.scoreThreshold[0]) {
        return 0
      }
      if (score >= this.scoreThreshold[1]) {
        return 1
      }
      return 2
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
    handleEdit: function(index, row) {
      const promise = getFeedbackRecordByKpeRecordId(row.id).then(response => {
        const h = this.$createElement
        this.form2 = response.data
        console.log(this.form2)
        if (this.form2) {
          this.totalNum = response.data['totalnum']
          this.$notify({
            title: '搜索成功',
            message: h('i', { style: 'color: teal' }, '搜索成功'),
            duration: 1000
          })
          this.updateDialogVisible = true
        } else {
          this.$notify({
            title: '没有结果',
            message: h('i', { style: 'color: teal' }, '没有反馈记录'),
            duration: 1000
          })
        }
      }).catch(err => {
        console.log(err)
      })
    },
    handleDelete(index, row) {
      deleteKpeRecord(row.id).then(response => {
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
      done()
    },
    clearForm() {
      this.form = {
        word: '',
        weight: '',
        status: '2',
        add_date: (new Date()).toLocaleDateString()
      }
    }

  }
}
</script>

<style scoped>

</style>
