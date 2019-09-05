import scrapy

# 如何使用
# scrapy runspider 04_scrapy_spider.py
# https://docs.scrapy.org/en/latest/intro/tutorial.html


class WdxtubSpider(scrapy.Spider):
    name = 'wdxtub_spider'
    start_urls = ['https://wdxtub.com']

    def parse(self, response):
        print(response.body)
        # for title in response.css('.post-header>h2'):
        #     yield {'title': title.css('a ::text').get()}
        #
        # for next_page in response.css('a.next-posts-link'):
        #     yield response.follow(next_page, self.parse)
