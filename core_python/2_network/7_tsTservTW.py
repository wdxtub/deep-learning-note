# 一个时间戳 TCP 服务器，使用 Twisted Internet 类开发
from twisted.internet import protocol, reactor
from time import ctime

PORT = 8778


class TSServProtocol(protocol.Protocol):
    def connectionMade(self):
        self.clnt = self.transport.getPeer().host
        clnt = self.clnt
        print('连接来自', clnt)

    def dataReceived(self, data):
        message = '[%s] %s' % (ctime(), data)
        self.transport.write(message.encode())


factory = protocol.Factory()
factory.protocol = TSServProtocol
print('TCP 服务器等待连接...')
reactor.listenTCP(PORT, factory)
reactor.run()
