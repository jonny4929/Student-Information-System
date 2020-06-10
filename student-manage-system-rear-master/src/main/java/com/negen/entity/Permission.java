package com.negen.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;

/**
 * @ Author     ：Negen
 * @ Date       ：Created in 17:48 2020/3/5
 * @ Description：权限实体
 * @ Modified By：
 * @Version: 1.0
 */
@Entity
@Table(name = "tb_permission")
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Permission {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    long id;
    String permissionName;	//权限名称

    public Permission(String permissionName){
        this.permissionName = permissionName;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getPermissionName() {
        return permissionName;
    }

    public void setPermissionName(String permissionName) {
        this.permissionName = permissionName;
    }
}
