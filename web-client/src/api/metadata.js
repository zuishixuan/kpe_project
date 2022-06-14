import request from '@/utils/request'

export function getPage(pageinfo) {
  return request({
    url: '/metadata/page',
    method: 'get',
    params: pageinfo
  })
}

export function save(form) {
  return request({
    url: '/metadata/save',
    method: 'post',
    data: form
  })
}

export function getInfo(name) {
  return request({
    url: '/user/info/' + name,
    method: 'get'
  })
}

export function logout() {
  return request({
    url: '/user/logout',
    method: 'get'
  })
}

export function getEsPage(pageinfo) {
  return request({
    url: '/es-metadata/page',
    method: 'get',
    params: pageinfo
  })
}

export function getEsMatchPage(searchinfo) {
  return request({
    url: '/es-metadata/match-page',
    method: 'get',
    params: searchinfo
  })
}
