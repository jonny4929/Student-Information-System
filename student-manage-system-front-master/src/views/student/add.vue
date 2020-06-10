<template>
  <el-form ref="form" :model="form" label-width="80px" style="width:50%;margin-top:20px">
    <el-form-item label="姓名">
      <el-input v-model="form.name" />
    </el-form-item>
    <el-form-item label="年龄">
      <el-input v-model="form.age" />
    </el-form-item>
    <el-form-item label="性别">
      <el-radio-group v-model="form.sex">
        <el-radio label="1" value="1">男</el-radio>
        <el-radio label="0" value="0">女</el-radio>
      </el-radio-group>
    </el-form-item>
    <el-form-item label="学号">
      <el-input v-model="form.num" />
    </el-form-item>
    <el-form-item label="年级">
      <el-select v-model="form.grade" placeholder="请选择年级">
        <el-option
          v-for="item in options"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        />
      </el-select>
    </el-form-item>
    <el-collapse-transition>
      <el-form-item v-show="classVisible" label="班级">
        <el-select v-model="form.clazz" placeholder="请选择班级">
          <el-option
            v-for="item in clazzList"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </el-form-item>
    </el-collapse-transition>
    <el-form-item label="家庭住址">
      <el-input v-model="form.address" />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" @click="onSubmit">添加</el-button>
      <el-button>取消</el-button>
    </el-form-item>
  </el-form>
</template>
<script>
import { getGrades, getClazzs } from '@/api/clazz'
import { addStudent } from '@/api/student'
export default {
  data() {
    return {
      options: [{
        value: '大一',
        label: '大一'
      }, {
        value: '大二',
        label: '大二'
      }, {
        value: '大三',
        label: '大三'
      }, {
        value: '大四',
        label: '大四'
      }],
      classVisible: true,
      clazzList: [{
        value: '一班',
        label: '一班'
      }, {
        value: '二班',
        label: '二班'
      }, {
        value: '三班',
        label: '三班'
      }, {
        value: '四班',
        label: '四班'
      }],
      grade: '',
      form: {
        name: '',
        age: '',
        sex: '',
        num: '',
        clazz: '',
        address: ''
      }
    }
  },
  watch: {
    grade: {
      handler: function(newVal, oldVal) {
        // eslint-disable-next-line eqeqeq
        if (newVal != oldVal) {
          getClazzs(newVal).then(
            response => {
              this.clazzList = response.data
            }
          )
          this.classVisible = true
        }
      },
      deep: true
    }
  },
  mounted: function() {
    // 加载年级数据
    getGrades().then(
      response => {
        this.gradeList = response.data
      }
    )
  },
  methods: {
    onSubmit() {
      // 提交添加请求
      var name = this.form.name
      var age = this.form.age
      var sex = this.form.sex
      var num = this.form.num
      var grade = this.form.grade
      var clazz = this.form.clazz
      var address = this.form.address
      if (name === null || name === '' ||
           age === null || age === '' ||
           sex === null || sex === '' ||
           num === null || num === '' ||
           grade === null || grade === '' ||
           clazz === null || clazz === '' ||
           address === null || address === ''
      ) {
        this.$message({
          message: '请填写完整的信息',
          type: 'error'
        })
      } else {
        var data = {
          name: name,
          age: age,
          sex: sex,
          num: num,
          grade: grade,
          clazz: clazz,
          address: address
        }
        addStudent(data).then(
          response => {
            this.$message({
              message: response.message,
              type: 'success'
            })
            this.form.name = ''
            this.form.age = ''
            this.form.sex = ''
            this.form.num = ''
            // this.grade = "";
            this.form.clazz = ''
            this.form.address = ''
          }
        )
      }
    }
  }
}
</script>
