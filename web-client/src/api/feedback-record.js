import request from '@/utils/request'

export function getPage(pageinfo) {
  return request({
    url: '/feedback-record/page',
    method: 'get',
    params: pageinfo
  })
}

export function saveFeedbackRecord(form) {
  return request({
    url: '/feedback-record/save',
    method: 'post',
    data: form
  })
}

export function deleteFeedbackRecord(id) {
  return request({
    url: '/feedback-record/delete/' + id,
    method: 'get'
  })
}

export function updateFeedbackRecord(form) {
  return request({
    url: '/feedback-record/update',
    method: 'post',
    data: form
  })
}

export function getFeedbackRecordByKpeRecordId(kpe_record_id) {
  return request({
    url: '/feedback-record/get-by-kpe-record-id/' + kpe_record_id,
    method: 'get'
  })
}

