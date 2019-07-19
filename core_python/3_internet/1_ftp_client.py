# 一个简单的 FTP 客户端

import ftplib
import os
import socket

HOST = 'ftp.mozilla.org'
DIRN = 'pub/mozilla.org/webtools'
FILE = 'bugzilla-LATEST.tar.gz'


def main():
    try:
        f = ftplib.FTP(HOST)
    except (socket.error, socket.gaierror) as e:
        print('ERROR: 无法连接 "{}"'.format(HOST))
        return
    print('*** 已连接到 "{}"'.format(HOST))

    try:
        f.login()
    except ftplib.error_perm:
        print('ERROR: 无法匿名登录')
        f.quit()
        return
    print('*** 已匿名身份登录')

    try:
        f.cwd(DIRN)
    except ftplib.error_perm:
        print('ERROR: 无法跳转到 "{}" 目录'.format(DIRN))
        f.quit()
        return
    print('*** 跳转到 "{}" 目录'.format(DIRN))

    try:
        f.retrbinary('RETR %s' % FILE, open(FILE, 'wb').write)
    except ftplib.error_perm:
        print('ERROR: 无法读取文件 "{}"'.format(FILE))
        os.unlink(FILE)
    else:
        print('*** 已下载 "{}" 到当前目录'.format(FILE))
    f.quit()


if __name__ == '__main__':
    main()
