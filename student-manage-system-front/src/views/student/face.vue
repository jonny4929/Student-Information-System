<template>
  <div class="app-container" >
    <el-form ref="form" :model="form" label-width="120px" style="width:50%;margin-top:20px" >
      <el-form-item label="头像">
        <el-upload
          class="upload-demo"
          drag
          action="https://jsonplaceholder.typicode.com/posts/"
          multiple>
          <i class="el-icon-upload"></i>
          <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
          <div class="el-upload__tip" slot="tip">只能上传jpg/png文件，且不超过500kb</div>
        </el-upload>
        <el-row>
          <el-button @click="test()" type="primary">识别签到</el-button>
        </el-row>
      </el-form-item>
    </el-form>
  </div>
</template>

<script>
  import GLOBAL from './global'
  export default {

    methods: {
      test() {
        if((GLOBAL.iii%3)===0)
        {
          alert('2018210421 签到成功');
          GLOBAL.iii+=1;
        }
        else if((GLOBAL.iii%3)===1){
          alert('2018210427 签到成功');
          GLOBAL.iii+=1;
        }
        else{
          alert('无匹配结果');
          GLOBAL.iii+=1;
        }
      },
    name: 'face'
  }}
</script>

<style scoped>

</style>
