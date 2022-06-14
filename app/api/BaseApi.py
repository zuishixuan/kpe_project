from app.common import HTTPClient
from config.config import settings
from requests.auth import HTTPBasicAuth

USER_FORM = {
    'name': 'esdata',
    'pass': 'EsData123654'
}


class BaseApi:

    def __init__(self):
        self._client = HTTPClient(settings['ES_URI'])

    def get_page(self, pageno, pagesize):
        results = self._client.get('/patent/page?current=' + pageno + '&size=' + pagesize,
                                   auth=HTTPBasicAuth(USER_FORM['name'], USER_FORM['pass']))
        return results
