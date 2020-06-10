package com.negen.entity;

import lombok.Data;
import net.bytebuddy.implementation.bind.annotation.Super;

import javax.persistence.*;

/**
 * @ Author     ：Negen
 * @ Date       ：Created in 15:34 2020/3/6
 * @ Description：教师实体
 * @ Modified By：
 * @Version: 1.0
 * 	姓名、年龄、性别、工号、科目
 */
@Data
@Table(name = "tb_teacher")
@Entity
public class Teacher {
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public Integer getSex() {
        return sex;
    }

    public void setSex(Integer sex) {
        this.sex = sex;
    }

    public String getNum() {
        return num;
    }

    public void setNum(String num) {
        this.num = num;
    }

    public String getCourse() {
        return course;
    }

    public void setCourse(String course) {
        this.course = course;
    }

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    Long id;
    String name;
    Integer age;
    Integer sex;
    String num;
    String course;
}
