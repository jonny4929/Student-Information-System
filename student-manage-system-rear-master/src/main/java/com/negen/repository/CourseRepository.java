package com.negen.repository;

import com.negen.entity.Course;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface CourseRepository extends JpaRepository<Course, Long> {
    List<Course> findByNameContaining(String name);

    List<Course> findByNumContaining(String content);
}
