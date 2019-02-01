# coding: utf-8
import re
import urllib.parse

from bs4 import BeautifulSoup


class HTMLParser(object):

    def parse(self, new_url, html_cont):
        if new_url is None or html_cont is None:
            return
        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        new_urls = self._get_new_urls(new_url, soup)
        new_data = self._get_new_data(new_url, soup)
        # print("parse is successfully called")
        return new_urls, new_data

    def _get_new_data(self, new_url, soup):
        res_data = {}
        res_data['url'] = new_url
        title_node = soup.find('dd', class_="lemmaWgt-lemmaTitle-title").find("h1")
        res_data['title'] = title_node.get_text()
        summary_node = soup.find('div', class_="lemma-summary")
        res_data['summary'] = summary_node.get_text()
        # <dd class="lemmaWgt-lemmaTitle-title">
        # <h1>Python</h1>
        # <div class="lemma-summary" label-module="lemmaSummary">
        # <div class="para" label-module="para">
        print("_get_new_data is successfully called")
        # print(res_data)
        return res_data

    def _get_new_urls(self, new_url, soup):
        new_urls = set()
        links = soup.find_all('a', href=re.compile(r'/item/+'))
        for link in links:
            url = link['href']
            print(url)
            new_full_url = urllib.parse.urljoin(new_url, url)
            new_urls.add(new_full_url)
        print("_get_new_urls is successfully called")
        return new_urls
