<template>
  <el-form ref="form" :model="form" label-width="80px" style="width:50%;margin-top:20px" >
    <el-form-item label="课程名">
      <el-input v-model="form.name"></el-input>
    </el-form-item>
    <el-form-item label="老师">
      <el-input v-model="form.teacher"></el-input>
    </el-form-item>
    <el-form-item label="上课教室">
      <el-input v-model="form.date"></el-input>
    </el-form-item>
    <el-form-item label="时间">
      <el-input v-model="form.time"></el-input>
    </el-form-item>

    <el-form-item>
      <el-button type="primary" @click="onSubmit">添加</el-button>
      <el-button>取消</el-button>
    </el-form-item>
  </el-form>
</template>
<script>
  import {addCourse} from '@/api/course'
  export default {
    data() {
      return {
        classVisible:false,
        form: {
          name: '',
          teacher: '',
          date: '',
          time: ''
        }
      }
    },
    methods: {
      onSubmit() {
        var name = this.form.name;
        var teacher = this.form.teacher;
        var date = this.form.date;
        var time = this.form .time;
        var data = {
          name: name,
          teacher: teacher,
          date: date,
          time: time
        }

        addCourse(data).then(
          response => {
            this.$message({
              message: response.message,
              type: "success"
            })
          }
        )

      }
    },
    // watch:{
    //   grade: {
    //     handler: function(newVal, oldVal){
    //       if(newVal != oldVal){
    //         this.classVisible = true
    //       }
    //     },
    //     deep: true
    //   }
    // }
  }
</script>
