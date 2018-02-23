# -*- coding: utf-8 -*-
import scrapy


class HandbookSpider(scrapy.Spider):
    name = 'handbook'
    allowed_domains = ['https://about.gitlab.com/handbook/']
    start_urls = ['http://https://about.gitlab.com/handbook//']

    def parse(self, response):
        pass
