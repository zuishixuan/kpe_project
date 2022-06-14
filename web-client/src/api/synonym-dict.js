import request from '@/utils/request'

export function getPage(pageinfo) {
  return request({
    url: '/synonym-dict/page',
    method: 'get',
    params: pageinfo
  })
}

export function saveSynonym(form) {
  return request({
    url: '/synonym-dict/save',
    method: 'post',
    data: form
  })
}

export function deleteSynonym(id) {
  return request({
    url: '/synonym-dict/delete/' + id,
    method: 'get'
  })
}

export function updateSynonym(form) {
  return request({
    url: '/synonym-dict/update',
    method: 'post',
    data: form
  })
}
