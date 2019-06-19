import os
from keras.datasets import mnist
from autokeras.image.image_supervised import ImageClassifier
from autokeras.utils import pickle_from_file
from graphviz import Digraph


def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
        to_pdf(graph, os.path.join(path, str(model_id)))


if __name__ == '__main__':
    # 需要把数据放到 ~/.keras/dataset 中，不然下载会报错
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    # (60000, 28, 28)
    print('增加一个维度，以符合格式要求')
    x_train = x_train.reshape(x_train.shape + (1,))
    print(x_train.shape)
    # (60000, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape + (1,))

    # 指定模型更新路径
    clf = ImageClassifier(path="automodels/", verbose=True)
    # 限制为 4 个小时
    # 搜索部分
    gap = 6
    clf.fit(x_train[::gap], y_train[::gap], time_limit=4*60*60)
    # 用表现最好的再训练一次
    clf.final_fit(x_train[::gap], y_train[::gap], x_test, y_train, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y)

    print("导出训练好的模型")
    clf.export_autokeras_model("automodels/auto_mnist_model")
    print("可视化模型")
    visualize("automodels/")
