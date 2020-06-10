<template>
  <div class="">
    <el-table
      :data="tableData"
      border
      style="width: 100%">
      <el-table-column
        label="课程序号"
        width=""
        align="center">
        <template slot-scope="scope">
          {{ scope.row.id }}
        </template>
      </el-table-column>
      <el-table-column
        label="课程名"
        width=""
        align="center">
        <template slot-scope="scope">
          {{ scope.row.name }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="课程教师"
        width=""
        align="center">
        <template slot-scope="scope">
          {{ scope.row.teacher }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="教室位置"
        width=""
        align="center">
        <template slot-scope="scope">
          {{ scope.row.date }}
        </template>
      </el-table-column>
      <el-table-column
        label="上课时间"
        width=""
        align="center">
        <template slot-scope="scope">
          {{ scope.row.time }}
          </el-popover>
        </template>
      </el-table-column>

      <el-table-column
        label="操作"
        align="center">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="primary"
            @click="handleEdit(scope.$index, scope.row)">编辑</el-button>
          <el-button
            size="mini"
            type="danger"
            @click="handleDelete(scope.$index, scope.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>


    <el-dialog title="编辑学生信息" :visible.sync="dialogFormVisible">
      <el-form :model="form">
        <el-input v-model="form.index" autocomplete="off" v-show="false"></el-input>
        <el-input v-model="form.id" autocomplete="off" v-show="false"></el-input>
        <el-form-item label="课程名" :label-width="formLabelWidth">
          <el-input v-model="form.name" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="授课教师" :label-width="formLabelWidth">
          <el-input v-model="form.teacher" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="教室位置" :label-width="formLabelWidth">
          <el-input v-model="form.date" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="上课时间" :label-width="formLabelWidth">
          <el-input v-model="form.time" autocomplete="off"></el-input>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="dialogFormVisible = false">取 消</el-button>
        <el-button type="primary" @click="handleUpdateStudent()">确 定</el-button>
      </div>
    </el-dialog>
  </div>
</template>




<script type="text/javascript">
  import {listCourse, updateCourse, deleteCourse} from '@/api/course'
  export default {
    data () {
      return {
        formLabelWidth:'',
        tableData: [],
        dialogFormVisible: false,
        form: {
          index: '',
          id: '',
          name: '',
          teacher: '',
          date: '',
          time: ''
        }

      }
    },
    methods: {
      handleEdit(index, row) {
        this.form.index = index;
        this.form.id = row.id
        this.form.name = row.name
        this.form.teacher = row.teacher
        this.form.date = row.date
        this.form.time = row.time
        this.dialogFormVisible = true;
      },
      handleDelete(index, row) {
        //删除记录
        var id = row.id;
        this.$confirm('此操作将永久删除该条数据, 是否继续?', '警告', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {
          deleteCourse(id).then(
            response => {
              this.tableData.splice(index, 1);
              this.$message({
                message: response.message,
                type: 'success'
              });
            }
          )
        }).catch((response) => {
          console.log(response);
          this.$message({
            type: 'info',
            message: '已取消删除'
          });
        });
      },
      loadData(){
        listCourse().then(
          response => {
            this.tableData = response.data;
          }
        )
      },
      handleUpdateStudent(){
        var id = this.form.id;
        var name = this.form.name;
        var teacher = this.teacher;
        var date = this.form.date;
        var time = this.form.time;
        var data = {
          id: id,
          name: name,
          teacher: teacher,
          date: date,
          time: time
        }

        updateCourse(data).then(
          //更新页面数据

          response => {
            // var i = this.form.index;
            // this.tableData[i].age = age;
            // this.tableData[i].sex = sex;
            // this.tableData[i].num = num;
            // this.tableData[i].name = name;
            // this.tableData[i].grade = grade;
            // this.tableData[i].clazz = clazz;
            // this.tableData[i].address = address;
            // this.dialogFormVisible = false;
            this.dialogFormVisible = false;
            this.loadData()
            this.$message({
              message: response.message,
              type: "success"
            })
          }
        )

      }
    },
    mounted: function(){
      this.loadData();
    }

  }
</script>
