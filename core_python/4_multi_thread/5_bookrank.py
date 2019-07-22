from atexit import register
from re import compile
from threading import Thread
from time import ctime
import requests

REGEX = compile('#([\d,]+) in Books ')
AMZN = 'http://amazon.com/dp/'
ISBNs = {
    '0132269937': 'Core Python Programming',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals'
}


# 会被 Amazon 的反爬禁止
def getRanking(isbn):
    page = requests.get('{}{}'.format(AMZN, isbn))
    return str(REGEX.findall(page.text)[0], 'utf-8')


def _showRanking(isbn):
    print('- {} ranked {}'.format(ISBNs[isbn], getRanking(isbn)))


def main():
    print('At {} on Amazon...'.format(ctime()))
    for isbn in ISBNs:
        # use multi thread
        Thread(target=_showRanking, args=(isbn,)).start()


@register
def _atexit():
    print('all DONE at:{}'.format(ctime()))


if __name__ == '__main__':
    main()
