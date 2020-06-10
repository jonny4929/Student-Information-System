package com.negen.service;

import com.alibaba.fastjson.JSONObject;
import com.negen.common.ServerResponse;
import com.negen.entity.Course;

public interface ICourseService {

    //添加教师信息
    ServerResponse addCourse(Course course);

    //修改教师信息
    ServerResponse updateCourse(Course course);

    //查看教师
    ServerResponse listCourse();

    //删除教师
    ServerResponse deleteCourse(Long id);

    //查询教师
    ServerResponse searchCourse(JSONObject obj);
}
