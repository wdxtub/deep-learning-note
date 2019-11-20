def map_fun(args, ctx):
    from datetime import datetime
    import tensorflow as tf
    import os
    import time

    import adanet
    from adanet.examples import simple_dnn

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    message = 'worker_num: {0}, job_name: {1}, task_index: {2}'.format(worker_num, job_name, task_index)

    input_dim = int(args.input_dim)
    batch_size = args.batch_size

    # Fix Random Seed
    RANDOM_SEED = 42

    FEATURES_KEY = "ctr"

    NUM_CLASSES = 2

    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    # head = tf.contrib.estimator.multi_class_head(NUM_CLASSES, loss_reduction=loss_reduction)
    head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)

    # numeric_column do not support SparseTensor
    feature_columns = [
        tf.feature_column.numeric_column(key=FEATURES_KEY, shape=[input_dim])
    ]

    log_dir = ctx.absolute_path(args.log_dir)
    export_dir = ctx.absolute_path(args.export_dir)
    print("tensorflow log path: {0}".format(log_dir))
    print("tensorflow export path: {0}".format(export_dir))

    def generator(ln):
        splits = tf.string_split([ln], delimiter=' ')
        label = splits.values[0]
        label = tf.string_to_number(label, tf.float64)
        label = tf.cond(label >= 1.0,
                        lambda: tf.constant(1, shape=[1], dtype=tf.float32),
                        lambda: tf.constant(0, shape=[1], dtype=tf.float32))

        # SparseTensor output
        col_val = tf.string_split(splits.values[1::], delimiter=':')
        col = tf.string_to_number(col_val.values[0::2], tf.int64) - 1

        vals = col_val.values[1::2]
        vals = tf.string_to_number(vals, tf.float32)

        # Filter the features which occurs few than given input_dim
        vals = tf.boolean_mask(vals, col < input_dim)
        col = tf.boolean_mask(col, col < input_dim)

        row = tf.cast(tf.fill(tf.shape(col), 0), tf.int64, name='row_cast')
        row_col = tf.transpose(tf.stack([row, col]), name='row_col_transpose')

        sparse = tf.SparseTensor(row_col, vals, (1, input_dim))

        # convert to dense，191106 必须转
        features = {FEATURES_KEY: tf.sparse_tensor_to_dense(sparse)}
        

        return features, label
    
    def new_input_fn(partition):
        def _input_fn():
            num_workers = len(ctx.cluster_spec['worker'])

            # path is ok
            data_dir = ctx.absolute_path(args.data_dir)
            file_pattern = os.path.join(data_dir, 'part-*')
            ds = tf.data.Dataset.list_files(file_pattern)
            ds = ds.shard(num_workers, task_index).repeat(args.epochs)

            if args.format == 'libsvm':
                ds = ds.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=10))
                parse_fn = generator

            if partition == "train":
                ds = ds.map(parse_fn, num_parallel_calls=5).shuffle(batch_size * 5)
            else:
                ds = ds.map(parse_fn, num_parallel_calls=5)

            #ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            return ds.batch(batch_size)

        return _input_fn

    
    print("========= Start Training")
    # just test
    LEARNING_RATE = 0.01
    TRAIN_STEPS = 1000
    ADANET_ITERATIONS = 2

    logdir = ctx.absolute_path(args.log_dir)

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=500,
        tf_random_seed=RANDOM_SEED,
        model_dir=logdir
    )

    # BaseLine Linear
    # estimator = tf.estimator.LinearClassifier(
    #     feature_columns=feature_columns,
    #     n_classes=NUM_CLASSES,
    #     optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
    #     loss_reduction=loss_reduction,
    #     config=config
    # )

    # DNN TEST - ADANET
    estimator = adanet.Estimator(
        head=head,
        subnetwork_generator=simple_dnn.Generator(
            layer_size=128,
            initial_num_layers=3,
            dropout=0.2,
            feature_columns=feature_columns,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=TRAIN_STEPS,
        evaluator=adanet.Evaluator(
            input_fn=new_input_fn("train"),
            steps=None
        ),
        config=config
    )

    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=new_input_fn("train"),
            max_steps=TRAIN_STEPS*3),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=new_input_fn("test"),
            steps=None)
    )



    if results is not None:
        print('result-------------------')
        print(message)
        print("Accuracy:", results["accuracy"])
        print("Loss:", results["average_loss"])
        message = "Accuracy: {}; Loss: {}".format(results["accuracy"], results["average_loss"])
    else:
        message = "No RESULTS! SOMETHING WRONG\n"
    
    print("==============================================")
    print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

    # WORKAROUND for https://github.com/tensorflow/tensorflow/issues/21745
    # wait for all other nodes to complete (via done files)
    done_dir = "{}/{}/done".format(ctx.absolute_path(args.log_dir), args.mode)
    print("Writing done file to: {}".format(done_dir))
    tf.gfile.MakeDirs(done_dir)
    with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
        done_file.write(message)

    for i in range(30):
        if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
            print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
            time.sleep(1)
        else:
            print("{} All nodes done".format(datetime.now().isoformat()))
            break