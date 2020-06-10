package com.negen.controller;

import com.alibaba.fastjson.JSONObject;
import com.negen.common.ServerResponse;
import com.negen.entity.Course;
import com.negen.service.ICourseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @ Author     ：Negen
 * @ Date       ：Created in 12:13 2020/3/12
 * @ Description：教师控制层
 * @ Modified By：
 * @Version: 1.0
 */
@RestController
@RequestMapping("course")
public class CourseController {
    @Autowired
    ICourseService courseService;
    //添加教师
    @RequestMapping(value = "add", method = RequestMethod.POST)
    public ServerResponse addCourse(@RequestBody Course course){
        return courseService.addCourse(course);
    }

    //修改教师
    @RequestMapping(value = "update", method = RequestMethod.POST)
    public ServerResponse updateCourse(@RequestBody Course course){
        return courseService.updateCourse(course);
    }

    //查看教师
    @RequestMapping(value = "list", method = RequestMethod.GET)
    public ServerResponse listCourse(){
        return courseService.listCourse();
    }
    //删除教师
    @RequestMapping(value = "delete/{id}", method = RequestMethod.POST)
    public ServerResponse deleteCourse(@PathVariable long id){
        return courseService.deleteCourse(id);
    }
    //查询教师
    @RequestMapping(value = "search", method = RequestMethod.POST)
    public ServerResponse searchCourse(@RequestBody JSONObject jsonObj){
        return courseService.searchCourse(jsonObj);
    }

}
