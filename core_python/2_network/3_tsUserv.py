# 创建一个 UDP 服务器，接收客户端发来的消息

from socket import *
from time import ctime

HOST = ''
PORT = 8778
BUFSIZ = 1024
ADDR = (HOST, PORT)

print("启动 UDP 服务端...")

udpSerSock = socket(AF_INET, SOCK_DGRAM)
udpSerSock.bind(ADDR)

try:
    while True:
        print('等待连接...')
        data, addr = udpSerSock.recvfrom(BUFSIZ)
        message = '[%s] %s' % (bytes(ctime(), 'utf-8'), data)
        udpSerSock.sendto(message.encode(), addr)
        print('连接来自:', addr)
except KeyboardInterrupt:
    print("CTRL-C Pressed..")
finally:
    udpSerSock.close()
