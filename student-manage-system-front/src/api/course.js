import request from '@/utils/request'

export function listCourse(){
  return request({
    url: '/course/list',
    method: 'get'
  })
}


export function addCourse(data){
  return request({
    url: '/course/add',
    method: 'post',
    data
  })
}


export function updateCourse(data){
  return request({
    url: '/course/update',
    method: 'post',
    data
  })
}


export function deleteCourse(id){
  return request({
    url: '/course/delete/' + id,
    method: 'post'
  })
}

export function searchCourse(data){
  return request({
    url: '/course/search',
    method: 'post',
    data
  })
}
