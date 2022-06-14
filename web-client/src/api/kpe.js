import request from '@/utils/request'

export function keywordExtract(data) {
  return request({
    url: '/kpe/extract',
    method: 'post',
    data
  })
}
