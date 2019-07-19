# 创建一个 TCP 服务器，接受来自客户端的请求
from socket import *
from time import ctime

HOST = ''
PORT = 8778
BUFSIZ = 1024
ADDR = (HOST, PORT)

print("启动 TCP 服务端...")

tcpSerSock = socket(AF_INET, SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

try:
    while True:
        print('等待连接...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('连接来自:', addr)
        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data:
                break
            message = '[%s] %s' % (bytes(ctime(), 'utf-8'), data)
            tcpCliSock.send(message.encode())
        tcpCliSock.close()
except KeyboardInterrupt:
    print("CTRL-C Pressed..")
finally:
    tcpSerSock.close()

