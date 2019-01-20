# coding: utf-8
import urllib.request


class HTMLDownloader(object):

    def download(self, new_url):
        if new_url is None:
            return None

        response = urllib.request.urlopen(new_url)

        if response.getcode() != 200:
            return None

        print("download is successfully called")
        return response.read()
