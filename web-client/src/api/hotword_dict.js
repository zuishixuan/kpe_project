import request from '@/utils/request'

export function getPage(timerange) {
  return request({
    url: '/search-record/search',
    method: 'get',
    params: timerange
  })
}
