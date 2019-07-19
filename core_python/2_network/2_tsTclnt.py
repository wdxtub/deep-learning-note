# TCP 时间戳客户端
from socket import *

HOST = '127.0.0.1'
PORT = 8778
BUFSIZ = 1024
ADDR = (HOST, PORT)

print("启动 TCP 客户端...")

tcpCliSock = socket(AF_INET, SOCK_STREAM)
tcpCliSock.connect(ADDR)

try:
    while True:
        data = input('> ')
        if not data:
            break
        tcpCliSock.send(data.encode())
        data = tcpCliSock.recv(BUFSIZ)
        if not data:
            break
        print(data.decode('utf-8'))
except KeyboardInterrupt:
    print("CTRL-C Pressed..")
finally:
    tcpCliSock.close()
