import request from '@/utils/request'

export function getPage(pageinfo) {
  return request({
    url: '/keyword-dict/page',
    method: 'get',
    params: pageinfo
  })
}

export function saveKeyword(form) {
  return request({
    url: '/keyword-dict/save',
    method: 'post',
    data: form
  })
}

export function deleteKeyword(id) {
  return request({
    url: '/keyword-dict/delete/' + id,
    method: 'get'
  })
}

export function updateKeyword(form) {
  return request({
    url: '/keyword-dict/update',
    method: 'post',
    data: form
  })
}

