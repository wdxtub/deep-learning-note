import scrapy

# 注意：该方法只是一个示例，并非完整的 scrapy 项目，详情参考官方教程文档
# 运行程序 scrapy runspider 04_scrapy_spider.py
# https://docs.scrapy.org/en/latest/intro/tutorial.html


class WdxtubSpider(scrapy.Spider):
    name = 'wdxtub_spider'
    # 抓取归档
    start_urls = ['https://wdxtub.com/archives']

    def parse(self, response):
        # 如何进行 parse，可以通过 scrapy shell url 进行探索，具体参考教程
        for header in response.css('article'):
            # 不需要单独用 print，会在屏幕上显示出来
            yield {
                'title': header.css('a span::text').get()
            }
        next_page = response.xpath('//a[has-class("extend next")]')
        link = next_page.xpath('@href').get()
        if link is not None:
            yield response.follow(link, self.parse)


