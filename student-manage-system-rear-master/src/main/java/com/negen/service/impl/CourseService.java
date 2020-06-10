package com.negen.service.impl;

import ch.qos.logback.classic.Logger;
import com.alibaba.fastjson.JSONObject;
import com.negen.common.ResponseEnum;
import com.negen.common.ServerResponse;
import com.negen.entity.Course;
import com.negen.repository.CourseRepository;
import com.negen.service.ICourseService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * @ Author     ：Negen
 * @ Date       ：Created in 13:52 2020/3/12
 * @ Description：教师业务类
 * @ Modified By：
 * @Version: 1.0
 */
@Service
@Slf4j
public class CourseService implements ICourseService {
    private static final String NUM = "1";
    private static final String NAME = "2";
    @Autowired
    CourseRepository courseRepository;

    @Override
    public ServerResponse addCourse(Course course) {
        try {
            courseRepository.save(course);
            return ServerResponse.getInstance().responseEnum(ResponseEnum.ADD_SUCCESS);
        }catch (Exception e){
            Logger log = null;
            log.info(e.getMessage());
            return ServerResponse.getInstance().responseEnum(ResponseEnum.INNER_ERROR);
        }
    }


    @Override
    public ServerResponse updateCourse(Course course) {
        try {
            long id = course.getId();
            Course _Course = courseRepository.getOne(id);
            _Course.setNum(course.getNum());
            _Course.setTeacher(course.getTeacher());
            _Course.setDate(course.getDate());
            _Course.setName(course.getName());
            _Course.setTime(course.getTime());
            courseRepository.save(_Course);
            return ServerResponse.getInstance().responseEnum(ResponseEnum.UPDATE_SUCCESS);
        }catch (Exception e){
            Logger log = null;
            log.info(e.getMessage());
            return ServerResponse.getInstance().responseEnum(ResponseEnum.INNER_ERROR);
        }
    }


    @Override
    public ServerResponse listCourse() {
        try {
            List<Course> courses = courseRepository.findAll();
            return ServerResponse.getInstance().responseEnum(ResponseEnum.GET_SUCCESS).data(courses);
        }catch (Exception e){
            Logger log = null;
            log.info(e.getMessage());
            return ServerResponse.getInstance().responseEnum(ResponseEnum.INNER_ERROR);
        }
    }

    @Override
    public ServerResponse deleteCourse(Long id) {
        try {
            courseRepository.deleteById(id);
            return ServerResponse.getInstance().responseEnum(ResponseEnum.DELETE_SUCCESS);
        }catch (Exception e){
            Logger log = null;
            log.info(e.getMessage());
            return ServerResponse.getInstance().responseEnum(ResponseEnum.INNER_ERROR);
        }
    }

    @Override
    public ServerResponse searchCourse(JSONObject obj) {
        try {
            String select = obj.getString("select");
            String content = obj.getString("content");
            List<Course> students = new ArrayList<>();
            switch (select){
                case NUM:
                    students = courseRepository.findByNumContaining(content);
                    break;
                case NAME:
                    students = courseRepository.findByNameContaining(content);
                    break;
                default:
                    break;
            }
            return ServerResponse.getInstance().responseEnum(ResponseEnum.GET_SUCCESS).data(students);

        }catch (Exception e){
            Logger log = null;
            log.info(e.getMessage());
            return ServerResponse.getInstance().responseEnum(ResponseEnum.INNER_ERROR);
        }
    }
}
