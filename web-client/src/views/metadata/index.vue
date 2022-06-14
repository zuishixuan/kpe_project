<template>
  <el-card class="box-card" shadow="hover">
    <el-form ref="form" :model="form" label-width="100px" style="margin-top: 30px">
      <el-form-item label="资源名称">
        <el-input v-model="form.name" placeholder="请输入资源名称" style="width: 700px" maxlength="50" show-word-limit />
      </el-form-item>
      <el-form-item label="标识符">
        <el-input v-model="form.identifier" placeholder="请输入资源标识符" style="width: 700px" />
      </el-form-item>
      <el-form-item label="学科分类">
        <el-input v-model="form.subject_category" placeholder="请输入资源学科分类，以英文分号分隔" style="width: 700px" />
      </el-form-item>
      <el-form-item label="主题分类">
        <el-input v-model="form.theme_category" placeholder="请输入资源主题" style="width: 700px" />
      </el-form-item>
      <el-form-item label="资源描述">
        <el-input v-model="form.describe" placeholder="请输入资源描述" type="textarea" style="width: 1000px" :rows="10" maxlength="1000" show-word-limit />
      </el-form-item>
      <!--    <el-form-item label="关键词">-->
      <!--      <el-input v-model="form.keyword" placeholder="请输入关键词，以英文分号分隔" style="width: 700px" />-->
      <!--    </el-form-item>-->
      <el-form-item>
        <el-button type="primary" @click="onExtract">生成关键词</el-button>
      </el-form-item>
      <el-form-item v-if="ex_flag" label="关键词候选" prop="type">
        <el-transfer v-model="form.keyword" :titles="['候选关键词', '选择关键词']" :data="keyword_be" />
      </el-form-item>
      <!--    <el-form-item label="资源生成日期">-->
      <!--      <el-col :span="11">-->
      <!--        <el-date-picker v-model="form.generate_date" type="date" placeholder="选择日期" style="width: 100%;" />-->
      <!--      </el-col>-->
      <!--    </el-form-item>-->
      <el-form-item>
        <el-button type="primary" @click="onSubmit">提交</el-button>
        <el-button>取消</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script>
import { keywordExtract } from '@/api/kpe'
import { save } from '@/api/metadata'

export default {
  data: function() {
    return {
      form: {
        name: '',
        identifier: '',
        subject_category: '',
        theme_category: '',
        describe: '',
        keyword: []
      },
      value: [],
      keyword_be: [],
      ex_flag: false
    }
  },
  'methods': {
    onSubmit() {
      this.form.keyword = this.form.keyword.join(';')
      save(this.form).then(response => {
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
    onExtract() {
      var data = {
        title: this.form.name,
        desc: this.form.describe
      }
      keywordExtract(data).then(response => {
        const h = this.$createElement
        const keywords = response.data
        console.log(keywords)
        this.keyword_be = []
        for (var i = 0, len = keywords.length; i < len; i++) {
          this.keyword_be.push({
            key: keywords[i]['word'],
            label: keywords[i]['word'],
            diable: false })
        }
        this.ex_flag = true
        this.$notify({
          title: '抽取成功',
          message: h('i', { style: 'color: teal' }, '抽取成功'),
          duration: 1000
        })
      }).catch(err => {
        console.log(err)
      })
      console.log('submit!')
    }
  }
}
</script>

<style scoped>

</style>
