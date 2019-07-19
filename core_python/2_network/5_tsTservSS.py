# 通过使用 SocketServer, TCPServer 和 StreamRequestHandler 来创建一个服务器
from socketserver import (TCPServer as TCP, StreamRequestHandler as SRH)
from time import ctime

HOST = ''
PORT = 8778
ADDR = (HOST, PORT)

class MyRequestHandler(SRH):
    def handle(self):
        print('连接来自', self.client_address)
        message = '[%s] %s' % (ctime(), self.rfile.readline())
        self.wfile.write(message.encode())


tcpServ = TCP(ADDR, MyRequestHandler)
print('TCP 服务器等待连接...')
tcpServ.serve_forever()