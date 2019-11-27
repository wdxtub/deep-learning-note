def map_fun(args, ctx):
    from datetime import datetime
    import tensorflow as tf
    import os
    import time
    import json

    import adanet
    from adanet.examples import simple_dnn

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    message = "worker_num: {0}, job_name: {1}, task_index: {2}".format(
        worker_num, job_name, task_index
    )
    print(message)
    input_dim = int(args.input_dim)
    batch_size = args.batch_size
    # Fix Random Seed
    RANDOM_SEED = 42

    FEATURES_KEY = "features"

    NUM_CLASSES = 2

    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    # head = tf.contrib.estimator.multi_class_head(NUM_CLASSES, loss_reduction=loss_reduction)
    head = tf.contrib.estimator.binary_classification_head(
        loss_reduction=loss_reduction
    )

    # numeric_column do not support SparseTensor
    feature_columns = [
        tf.feature_column.numeric_column(key=FEATURES_KEY, shape=[input_dim])
    ]

    log_dir = ctx.absolute_path(args.log_dir)
    export_dir = ctx.absolute_path(args.export_dir)
    pred_dir = ctx.absolute_path(args.prediction_dir)
    print("tensorflow log path: {0}".format(log_dir))
    print("tensorflow export path: {0}".format(export_dir))
    print("tensorflow prediction path: {0}".format(pred_dir))

    def generator(ln):
        splits = tf.string_split([ln], delimiter=" ")
        label = splits.values[0]
        label = tf.string_to_number(label, tf.float64)
        label = tf.cond(
            label >= 1.0,
            lambda: tf.constant(1, shape=[1], dtype=tf.float32),
            lambda: tf.constant(0, shape=[1], dtype=tf.float32),
        )

        # SparseTensor output
        col_val = tf.string_split(splits.values[1::], delimiter=":")
        col = tf.string_to_number(col_val.values[0::2], tf.int64) - 1

        vals = col_val.values[1::2]
        vals = tf.string_to_number(vals, tf.float32)

        # Filter the features which occurs few than given input_dim
        vals = tf.boolean_mask(vals, col < input_dim)
        col = tf.boolean_mask(col, col < input_dim)

        row = tf.cast(tf.fill(tf.shape(col), 0), tf.int64, name="row_cast")
        row_col = tf.transpose(tf.stack([row, col]), name="row_col_transpose")

        sparse = tf.SparseTensor(row_col, vals, (1, input_dim))

        # convert to dense，191106 必须转
        features = {FEATURES_KEY: tf.sparse_tensor_to_dense(sparse)}

        return features, label

    def new_input_fn(partition, training):
        def _input_fn():
            # path is ok
            parse_fn = generator

            if partition == "train":
                data_dir = ctx.absolute_path(args.data_dir)
                file_pattern = os.path.join(data_dir, "part-*")
                ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)

                ds = ds.apply(
                    tf.contrib.data.parallel_interleave(
                        tf.data.TextLineDataset, cycle_length=10
                    )
                )
                ds = ds.map(parse_fn, num_parallel_calls=5)
                if training:
                    ds = ds.shuffle(batch_size * 5).repeat()
            else:
                data_dir = ctx.absolute_path(args.test_dir)
                file_pattern = os.path.join(data_dir, "part-*")
                ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)

                ds = ds.apply(
                    tf.contrib.data.parallel_interleave(
                        tf.data.TextLineDataset, cycle_length=10
                    )
                )
                ds = ds.map(parse_fn, num_parallel_calls=5)

            iterator = ds.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

            # ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            # return ds.batch(batch_size)

        return _input_fn

    print("========= Start Training")
    LEARNING_RATE = 0.01
    TRAIN_STEPS = 3000
    ADANET_ITERATIONS = 3  # AKA Boosting Iteration
    # 控制模型复杂度
    ADANET_LAMBDA = 0.1
    LEARN_MIXTURE_WEIGHTS = False



    #strategy = adanet.distributed.RoundRobinStrategy()

    # 191125 这里一定要设置
    tfc = json.dumps(
        {"cluster": ctx.cluster_spec, "task": {"type": job_name, "index": task_index}}
    )
    os.environ["TF_CONFIG"] = tfc

    # 191127 尝试不用 device_filter，用了 strategy 后会自动设置为 /job:ps，不需要时候手动设置
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=5000,
        tf_random_seed=RANDOM_SEED,
        model_dir=log_dir,
    )

    # config = tf.estimator.RunConfig(
    #     save_checkpoints_steps=5000,
    #     tf_random_seed=RANDOM_SEED,
    #     model_dir=logdir,
    #     session_config=tf.ConfigProto(
    #         log_device_placement=False, device_filters=["/job:ps"]
    #     ),
    # )

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
        force_grow=True,
        subnetwork_generator=simple_dnn.Generator(
            layer_size=128,
            initial_num_layers=2,
            dropout=0.2,
            feature_columns=feature_columns,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
            seed=RANDOM_SEED,
        ),
        adanet_lambda=ADANET_LAMBDA,
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
        #evaluator=adanet.Evaluator(input_fn=new_input_fn("test", False)),
        evaluator=adanet.Evaluator(input_fn=new_input_fn("test", False), steps=1000),
        config=config,
        #experimental_placement_strategy=strategy,
        # 记录 report，实际上没啥用
        #     report_materializer=adanet.ReportMaterializer(
        #         input_fn=new_input_fn("train", False),
        #     ),
    )

    # 尝试不 return 任何东西，只是计算
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=new_input_fn("train", True), max_steps=TRAIN_STEPS
        ),
        # 这里的 Eval 在分布式场景下，实际上并没有任何作用
        eval_spec=tf.estimator.EvalSpec(
            input_fn=new_input_fn("test", False),
            steps=None,
            start_delay_secs=1,
            throttle_secs=30,
        ),
    )

    # 最后一轮只训练，模型参数会保存到 model.ckpt，并不会再为下一轮去做准备

    # 参考 https://github.com/tensorflow/adanet/blob/master/adanet/core/estimator_test.py
    # line 2362 def test_export_saved_model_always_uses_replication_placement(self):
    def serving_input_receiver_fn():
        serialized_sample = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, input_dim], name='features')
        tensor_features = {'features': serialized_sample}
        return tf.estimator.export.ServingInputReceiver(features=tensor_features, receiver_tensors=serialized_sample)

    # 在 RoundRobinStrategy 下无法执行
    if ctx.job_name == "chief":
        # 进行预测，分别是测试和训练
        print('export test result')
        predictions = estimator.predict(new_input_fn("test", False))
        print('Writing Predictions to {}'.format(pred_dir))
        tf.gfile.MakeDirs(pred_dir)
        with tf.gfile.GFile("{}/test".format(pred_dir), 'w') as f:
            for pred in predictions:
                f.write(str(pred))
                f.write('\n')
        print('export train result')
        predictions = estimator.predict(new_input_fn("train", False))
        print('Writing Predictions to {}'.format(pred_dir))
        tf.gfile.MakeDirs(pred_dir)
        with tf.gfile.GFile("{}/train".format(pred_dir), 'w') as f:
            for pred in predictions:
                f.write(str(pred))
                f.write('\n')
        # 导出模型
        estimator.export_saved_model(export_dir, serving_input_receiver_fn, experimental_mode=tf.estimator.ModeKeys.PREDICT)

    # 这里只能在单机版使用，tfos 上无法执行


    # 以下是单机版本
    # results, _ = tf.estimator.train_and_evaluate(
    #     estimator,
    #     train_spec=tf.estimator.TrainSpec(
    #         input_fn=new_input_fn("train"),
    #         max_steps=TRAIN_STEPS),
    #     eval_spec=tf.estimator.EvalSpec(
    #         input_fn=new_input_fn("test"),
    #         steps=None)
    # )
    # if results is not None:
    #     print('result-------------------')
    #     print(message)
    #     print("Accuracy:", results["accuracy"])
    #     print("Loss:", results["average_loss"])
    #     message = "Accuracy: {}; Loss: {}".format(results["accuracy"], results["average_loss"])
    # else:
    #     message = "No RESULTS! SOMETHING WRONG\n"
