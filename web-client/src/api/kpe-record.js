import request from '@/utils/request'

export function getPage(pageinfo) {
  return request({
    url: '/kpe-record/page',
    method: 'get',
    params: pageinfo
  })
}

export function saveKpeRecord(form) {
  return request({
    url: '/kpe-record/save',
    method: 'post',
    data: form
  })
}

export function deleteKpeRecord(id) {
  return request({
    url: '/kpe-record/delete/' + id,
    method: 'get'
  })
}

export function updateKpeRecord(form) {
  return request({
    url: '/kpe-record/update',
    method: 'post',
    data: form
  })
}

