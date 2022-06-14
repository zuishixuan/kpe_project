import request from '@/utils/request'

export function login(data) {
  return request({
    url: '/user/login',
    method: 'post',
    data
  })
}

export function getInfo(token) {
  return request({
    url: '/user/info/' + token,
    method: 'get'
  })
}

export function logout() {
  return request({
    url: '/user/logout',
    method: 'get'
  })
}

export function getUserPage(pageinfo) {
  return request({
    url: '/user/page',
    method: 'get',
    params: pageinfo
  })
}

export function saveUser(form) {
  return request({
    url: '/user/save',
    method: 'post',
    data: form
  })
}

export function deleteUser(id) {
  return request({
    url: '/user/delete/' + id,
    method: 'get'
  })
}

export function updateUser(form) {
  return request({
    url: '/user/update',
    method: 'post',
    data: form
  })
}
