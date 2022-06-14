import requests


class HTTPClient(object):

    def __init__(self, base_url=None, timeout=None, **kwargs):
        self.base_url = base_url
        self.timeout = timeout
        self.kwargs = kwargs
        self.session = requests.Session()

        # request请求重试
        if self.kwargs.get('retry'):
            request_retry = requests.adapters.HTTPAdapaters(
                max_retries=self.kwargs['retry'])
            self.session.mount('https://', request_retry)
            self.session.mount('http://', request_retry)

    def _request_wrapper(self, method, api, **kwargs):
        url = self.base_url + api
        print(
            f"sending {method} request to {url} ...  kwargs is {repr(kwargs)}")

        res = self.session.request(method, url, **kwargs)
        if res.status_code != 200:
            raise Exception(f"Http status code is not 200, status code {res.status_code}, "
                            f"response is {res.content}")
        # 返回有可能不是json格式
        if 'text/html' in res.headers['Content-Type']:
            print(f"sending {method} request to {url} over ... response is "
                  f"{repr(res.content)}")
            return res.text
        else:
            print(f"sending {method} request to {url} over ... response is "
                  f"{repr(res.json())}")
            return res.json() or dict()

        # return self.session.request(method, url, **kwargs)

    def get(self, api, **kwargs):
        return self._request_wrapper('GET', api, **kwargs)

    def options(self, api, **kwargs):
        return self._request_wrapper('OPTIONS', api, **kwargs)

    def head(self, api, **kwargs):
        return self._request_wrapper('HEAD', api, **kwargs)

    def post(self, api, **kwargs):
        return self._request_wrapper('POST', api, **kwargs)

    def put(self, api, **kwargs):
        return self._request_wrapper('PUT', api, **kwargs)

    def patch(self, api, **kwargs):
        return self._request_wrapper('PATCH', api, **kwargs)

    def delete(self, api, **kwargs):
        return self._request_wrapper('DELETE', api, **kwargs)

    def __del__(self):
        try:
            if hasattr(self, "session"):
                self.session.close()
        except Exception as e:
            print(e)
