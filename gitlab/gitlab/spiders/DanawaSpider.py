# coding: utf-8
import scrapy

class DanawaSpider(scrapy.Spider):
    name= "danawa"
    def start_requests(self):
        urls=[
            "http://prod.danawa.com/list/?cate=112753"
            , "http://prod.danawa.com/list/?cate=112747"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        page= response.url.split("=")[-1]
        filename= 'danawa-%s.html'%page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('saved file %s'%filename)
