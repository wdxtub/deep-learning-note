import tensorflow as tf

# Example 1: session keeps track of the random state
c = tf.random.uniform([], -10, 10, seed=2)

with tf.compat.v1.Session() as sess:
    print(sess.run(c)) # >> 3.574932
    print(sess.run(c)) # >> -5.9731865

# Example 2: each new session will start the random state all over again.
c = tf.random.uniform([], -10, 10, seed=2)

with tf.compat.v1.Session() as sess:
    print(sess.run(c)) # >> 3.574932

with tf.compat.v1.Session() as sess:
    print(sess.run(c)) # >> 3.574932

# Example 3: with operation level random seed, each op keeps its own seed.
c = tf.random.uniform([], -10, 10, seed=2)
d = tf.random.uniform([], -10, 10, seed=2)

with tf.compat.v1.Session() as sess:
    print(sess.run(c)) # >> 3.574932
    print(sess.run(d)) # >> 3.574932

# Example 4: graph level random seed
tf.compat.v1.set_random_seed(2)
c = tf.random.uniform([], -10, 10)
d = tf.random.uniform([], -10, 10)

with tf.compat.v1.Session() as sess:
    print(sess.run(c)) # >> 9.123926
    print(sess.run(d)) # >> -4.5340395
