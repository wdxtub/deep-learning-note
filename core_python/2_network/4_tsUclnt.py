# UDP 的客户端

from socket import *

HOST = 'localhost'
PORT = 8778
BUFSIZ = 1024
ADDR = (HOST, PORT)

print("启动 UDP 客户端...")

udpCliSock = socket(AF_INET, SOCK_DGRAM)

try:
    while True:
        data = input('> ')
        if not data:
            break
        udpCliSock.sendto(data.encode(), ADDR)
        data, ADDR = udpCliSock.recvfrom(BUFSIZ)
        if not data:
            break
        print(data.decode('utf-8'))
except KeyboardInterrupt:
    print("CTRL-C Pressed..")
finally:
    udpCliSock.close()
