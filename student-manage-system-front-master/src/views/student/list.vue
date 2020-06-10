<template>
<div class="">
  <el-table
      :data="tableData"
      border
      style="width: 100%">
      <el-table-column
        label="序号"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.id }}
          </template>
      </el-table-column>
      <el-table-column
        label="学号"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.num }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="姓名"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.name }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="年龄"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.age }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="性别"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.sex|convertSex }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="年级"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.grade }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="班级"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.clazz }}
          </el-popover>
        </template>
      </el-table-column>
      <el-table-column
        label="家庭住址"
        width=""
        align="center">
          <template slot-scope="scope">
            {{ scope.row.address }}
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
        <el-form-item label="学号" :label-width="formLabelWidth">
          <el-input v-model="form.num" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="姓名" :label-width="formLabelWidth">
          <el-input v-model="form.name" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="年龄" :label-width="formLabelWidth">
          <el-input v-model="form.age" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="性别" :label-width="formLabelWidth">
          <el-radio-group v-model="formsex">
            <el-radio :label="1">男</el-radio>
            <el-radio :label="0">女</el-radio>
          </el-radio-group>
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
        <el-form-item label="家庭住址" :label-width="formLabelWidth">
          <el-input v-model="form.address" autocomplete="off"></el-input>
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
import { getGrades, getClazzs } from '@/api/clazz';
import { listStudent, updateStudent, deleteStudent } from '@/api/student';

export default {
  data () {
    return {
      options:[{
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
      formLabelWidth:'',
      dialogFormVisible: false,
      clazzVisible: false,
      tableData: [],
      formGrade:'',
      formsex: 1,
      form: {
        index: '',
        id: '',
        num: '',
        name: '',
        age: '',
        clazz: '',
        address: ''
      }
    }
  },
  methods: {
      handleEdit(index, row) {
        this.form.index = index;
        this.form.id = row.id
        this.form.age = row.age
        this.form.sex = row.sex
        this.form.num = row.num
        this.form.name = row.name
        this.form.clazz = row.clazz
        this.form.address = row.address
        getGrades().then(
          response => {
            this.options = response.data
          }
        )
        this.formsex = row.sex
        this.formGrade = row.grade
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
          deleteStudent(id).then(
            response => {
              this.tableData.splice(index, 1);
              this.$message({
                  message: response.message,
                  type: 'success'
                });
            }
          )
         }).catch(() => {
           this.$message({
             type: 'info',
             message: '已取消删除'
           });
         });

      },
      handleUpdateStudent(){
        var id = this.form.id;
        var age = this.form.age;
        var sex = this.form.sex;
        var num = this.form.num;
        var name = this.form.name;
        var grade = this.form.grade;
        var clazz = this.form.clazz;
        var address = this.form.address;
        if(null === id || "" === id ||
        null === age || "" === age ||
        null === sex || "" === sex ||
        null === num || "" === num ||
        null === name || "" === name ||
        null === grade || "" === grade ||
        null === clazz || "" === clazz ||
        null === address || "" === address){
          this.$message({
            message: "请填写完整的信息",
            type: "error"
          })
        }else{
          //请求后台
          var data = {
            id: id,
            age: age,
            sex: sex,
            num: num,
            name: name,
            grade: grade,
            clazz: clazz,
            address: address,
          }
          updateStudent(data).then(
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
      loadData(){
        listStudent().then(
          response=>{
            this.tableData = response.data;
          }
        )
      }
  },
  mounted: function(){
    this.loadData()
  },
  filters: {
    convertSex(sex){
      switch(sex){
        case 1:
          return "男"
          break
        case 0:
          return "女"
          break
      }
    }
  },
  watch: {
    formGrade: {
      handler: function(newVal, oldVal){
        if(newVal != oldVal){
          getClazzs(newVal).then(
            response => {
              this.clazzList = response.data
            }
          )
          this.clazzVisible = true
        }
      },
      deep: true
    }
  }
}
</script>
