import adanet
from adanet.examples import simple_dnn

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))


def map_fun_v2(args, ctx):
    from datetime import datetime
    import tensorflow as tf
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # Parameters
    IMAGE_PIXELS = 28
    hidden_units = 128
    # Fix Random Seed
    RANDOM_SEED = 42

    (x_train, y_train), (x_test, y_test) = (tf.keras.datasets.mnist.load_data())

    FEATURES_KEY = "images"

    NUM_CLASSES = 10

    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    head = tf.contrib.estimator.multi_class_head(NUM_CLASSES, loss_reduction=loss_reduction)

    feature_columns = [
        tf.feature_column.numeric_column(FEATURES_KEY, shape=[28, 28, 1])
    ]

    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)

    def generator(images, labels):
        """Returns a generator that returns image-label pairs."""

        def _gen():
            for image, label in zip(images, labels):
                yield image, label

        return _gen

    def preprocess_image(image, label):
        """Preprocesses an image for an `Estimator`."""
        image = image / 255.
        image = tf.reshape(image, [28, 28, 1])
        features = {FEATURES_KEY: image}
        return features, label

    def input_fn(partition, training):
        """Generate an input_fn for the Estimator."""

        def _input_fn():
            if partition == "train":
                dataset = tf.data.Dataset.from_generator(
                    generator(x_train, y_train), (tf.float32, tf.int32), ((28, 28), ()))
            else:
                dataset = tf.data.Dataset.from_generator(
                    generator(x_test, y_test), (tf.float32, tf.int32), ((28, 28), ()))

            if training:
                dataset = dataset.shuffle(10 * args.batch_size, seed=RANDOM_SEED).repeat()

            dataset = dataset.map(preprocess_image).batch(args.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

        return _input_fn

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default
        # 这里的日志都是看不到的
        message = ""
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            print("========= Start Training")
            LEARNING_RATE = 0.003
            TRAIN_STEPS = 5000
            BATCH_SIZE = 64
            ADANET_ITERATIONS = 2

            logdir = ctx.absolute_path(args.model)

            config = tf.estimator.RunConfig(
                save_checkpoints_steps=50000,
                save_summary_steps=50000,
                tf_random_seed=RANDOM_SEED,
                model_dir=logdir
            )

            # 先测试下线性模型
            # estimator = tf.estimator.LinearClassifier(
            #     feature_columns=feature_columns,
            #     n_classes=NUM_CLASSES,
            #     optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            #     loss_reduction=loss_reduction,
            #     config=config
            # )

            estimator = adanet.Estimator(
                head=head,
                subnetwork_generator=simple_dnn.Generator(
                    feature_columns=feature_columns,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
                    seed=RANDOM_SEED),
                max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
                evaluator=adanet.Evaluator(
                    input_fn=input_fn("train", training=False),
                    steps=None),
                config=config
            )

            results, _ = tf.estimator.train_and_evaluate(
                estimator,
                train_spec=tf.estimator.TrainSpec(
                    input_fn=input_fn("train", training=True),
                    max_steps=TRAIN_STEPS),
                eval_spec=tf.estimator.EvalSpec(
                    input_fn=input_fn("test", training=False),
                    steps=None)
            )

            print("Accuracy:", results["accuracy"])
            print("Loss:", results["average_loss"])
            message = "Accuracy: {}; Loss: {}".format(results["accuracy"], results["average_loss"])
            print("==============================================")

        print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

        # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
        # wait for all other nodes to complete (via done files)
        done_dir = "{}/{}/done".format(ctx.absolute_path(args.model), args.mode)
        print("Writing done file to: {}".format(done_dir))
        tf.gfile.MakeDirs(done_dir)
        with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
            done_file.write("done")
            done_file.write(message)

        for i in range(60):
            if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
                print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
                time.sleep(1)
            else:
                print("{} All nodes done".format(datetime.now().isoformat()))
                break


def map_fun(args, ctx):
    from datetime import datetime
    import math
    import numpy
    import tensorflow as tf
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # Parameters
    IMAGE_PIXELS = 28
    hidden_units = 128


    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)

    # Create generator for Spark data feed
    tf_feed = ctx.get_data_feed(args.mode == 'train')

    def rdd_generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) == 0:
                return
            row = batch[0]
            image = numpy.array(row[0]).astype(numpy.float32) / 255.0
            label = numpy.array(row[1]).astype(numpy.int64)
            yield (image, label)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # Dataset for input data
            ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (
            tf.TensorShape([IMAGE_PIXELS * IMAGE_PIXELS]), tf.TensorShape([10]))).batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()
            x, y_ = iterator.get_next()

            # Variables of the hidden layer
            hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                                                    stddev=1.0 / IMAGE_PIXELS), name="hid_w")
            hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
            tf.summary.histogram("hidden_weights", hid_w)

            # Variables of the softmax layer
            sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                                                   stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
            tf.summary.histogram("softmax_weights", sm_w)

            x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
            tf.summary.image("x_img", x_img)

            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)

            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

            global_step = tf.train.get_or_create_global_step()

            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
            tf.summary.scalar("loss", loss)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)

            # Test trained model
            label = tf.argmax(y_, 1, name="label")
            prediction = tf.argmax(y, 1, name="prediction")
            correct_prediction = tf.equal(prediction, label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            tf.summary.scalar("acc", accuracy)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = ctx.absolute_path(args.model)
        print("tensorflow model path: {0}".format(logdir))
        summary_writer = tf.summary.FileWriter("tensorboard_%d" % worker_num, graph=tf.get_default_graph())

        hooks = [tf.train.StopAtStepHook(last_step=args.steps)] if args.mode == "train" else []
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op,
                                                                          saver=saver),
                                               checkpoint_dir=logdir,
                                               hooks=hooks) as sess:
            print("{} session ready".format(datetime.now().isoformat()))

            # Loop until the session shuts down or feed has no more data
            step = 0
            while not sess.should_stop() and not tf_feed.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                if args.mode == "train":
                    _, summary, step = sess.run([train_op, summary_op, global_step])
                    if (step % 100 == 0) and (not sess.should_stop()):
                        print("{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
                    if task_index == 0:
                        summary_writer.add_summary(summary, step)
                else:  # args.mode == "inference"
                    labels, preds, acc = sess.run([label, prediction, accuracy])
                    results = ["{} Label: {}, Prediction: {}".format(datetime.now().isoformat(), l, p) for l, p in
                               zip(labels, preds)]
                    tf_feed.batch_results(results)
                    print("acc: {}".format(acc))

        print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

        if sess.should_stop() or step >= args.steps:
            tf_feed.terminate()

        # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
        # wait for all other nodes to complete (via done files)
        done_dir = "{}/{}/done".format(ctx.absolute_path(args.model), args.mode)
        print("Writing done file to: {}".format(done_dir))
        tf.gfile.MakeDirs(done_dir)
        with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
            done_file.write("done")
            done_file.write("good")

        for i in range(60):
            if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
                print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
                time.sleep(1)
            else:
                print("{} All nodes done".format(datetime.now().isoformat()))
                break
