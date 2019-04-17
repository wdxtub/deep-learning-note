from keras import layers
from keras import Input
from keras.models import Model

print("输入一个人的一系列社交发帖，尝试预测年龄、性别和收入水平")
vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediciton = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediciton, income_prediction, gender_prediction])
print("带权重的多重损失，这样能更好平衡多个任务")
output_has_name = True
if output_has_name:
    model.compile(optimizer='rmsprop', 
        loss={
            'age': 'mse', 
            'income': 'categorical_crossentropy', 
            'gender': 'binary_crossentropy'},
        loss_weights={
            'age': 0.25,
            'income': 1.,
            'gender': 10.
        })
else:
    model.compile(optimizer='rmsprop', 
        loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
        loss_weights=[0.25, 1., 10.])

print("假设 xxx_targets 为 Numpy 数组，把输入给到模型中")
if output_has_name:
    model.fit(posts, {
        'age': age_targets,
        'income': income_targets,
        'gender': gender_targets
    }, epochs=10, batch_size=64)
else:
    model.fit(posts, [age_targets, income_targets, gender_targets],
    epochs=10, batch_size=64)