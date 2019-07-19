from twisted.internet import protocol, reactor

HOST = 'localhost'
PORT = 8778

print("TCP 客户端启动中...")


class TSClntProtocol(protocol.Protocol):
    def sendData(self):
        data = input('> ')
        if data:
            print('发送数据...', data)
            self.transport.write(data.encode())
        else:
            self.transport.loseConnection()

    def connectionMade(self):
        self.sendData()

    def dataReceived(self, data):
        print(data)
        self.sendData()


class TSClntFactory(protocol.ClientFactory):
    protocol = TSClntProtocol
    clientConnectionLost = clientConnectionFailed = \
        lambda self, connecotr, reason: reactor.stop()


reactor.connectTCP(HOST, PORT, TSClntFactory())
reactor.run()