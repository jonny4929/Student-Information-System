package com.negen.vo;

import lombok.Data;

import javax.persistence.Lob;
import java.util.List;

/**
 * @ Author     ：Negen
 * @ Date       ：Created in 8:56 2020/3/7
 * @ Description：返回登录用户的详细信息
 * @ Modified By：
 * @Version: 1.0
 * {"code":20000,"data":{"roles":["admin"],"introduction":"I am a super administrator","avatar":"https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif","name":"Super Admin"}}
 */
@Data
public class UserInfoVo {
    Long id;
    String name;
    String avatar;

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

    public String getAvatar() {
        return avatar;
    }

    public void setAvatar(String avatar) {
        this.avatar = avatar;
    }

    public String getIntroduction() {
        return introduction;
    }

    public void setIntroduction(String introduction) {
        this.introduction = introduction;
    }

    public List getRoles() {
        return roles;
    }

    public void setRoles(List roles) {
        this.roles = roles;
    }

    @Lob
    String introduction;
    List roles;
}
