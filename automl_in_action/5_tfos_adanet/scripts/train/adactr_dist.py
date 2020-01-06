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

    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    def weighted_cross_entropy_with_logits(labels, logits):
        return tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=4)

    head = tf.contrib.estimator.binary_classification_head(
        loss_reduction=loss_reduction,
        loss_fn=weighted_cross_entropy_with_logits
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
    TRAIN_STEPS = 1000
    ADANET_ITERATIONS = 4  # AKA Boosting Iteration
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

    # estimator = tf.estimator.LinearEstimator(
    #     head=head,
    #     feature_columns=feature_columns,
    #     config=config
    #
    # )

    # config = tf.estimator.RunConfig(
    #     save_checkpoints_steps=5000,
    #     tf_random_seed=RANDOM_SEED,
    #     model_dir=logdir,
    #     session_config=tf.ConfigProto(
    #         log_device_placement=False, device_filters=["/job:ps"]
    #     ),
    # )



    # DNN TEST - ADANET
    estimator = adanet.Estimator(
        head=head,
        force_grow=False,
        subnetwork_generator=simple_dnn.Generator(
            layer_size=128,
            initial_num_layers=1,
            dropout=0.2,
            feature_columns=feature_columns,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
            seed=RANDOM_SEED,
        ),
        adanet_lambda=ADANET_LAMBDA,
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(input_fn=new_input_fn("test", False), steps=1000),
        config=config,
    )

    # ensemble_estimator = adanet.AutoEnsembleEstimator(
    #     head=head,
    #     candidate_pool= lambda config: {
    #         "linear1":
    #             tf.estimator.LinearEstimator(
    #                 head=head,
    #                 feature_columns=feature_columns,
    #                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1),
    #                 config=config,
    #             ),
    #         "dnn1":
    #             tf.estimator.DNNEstimator(
    #                 head=head,
    #                 feature_columns=feature_columns,
    #                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
    #                 hidden_units=[512, 256, 128],
    #                 config=config,
    #             ),
    #         "dnn2":
    #             tf.estimator.DNNEstimator(
    #                 head=head,
    #                 feature_columns=feature_columns,
    #                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
    #                 hidden_units=[256, 128],
    #                 config=config,
    #             ),
    #         "dnn_linear":
    #             tf.estimator.DNNLinearCombinedEstimator(
    #                 head=head,
    #                 dnn_feature_columns=feature_columns,
    #                 linear_feature_columns=feature_columns,
    #                 dnn_hidden_units=[512, 256, 128],
    #                 config=config,
    #             )
    #     },
    #     max_iteration_steps=100,
    # )

    cur_e = estimator

    # 尝试不 return 任何东西，只是计算
    tf.estimator.train_and_evaluate(
        cur_e,
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
    # 这样的保存方式，需要输入是一个 example，不适合 DSP 的输入
    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    def serving_input_receiver_fn():
        indices = tf.placeholder(dtype=tf.int64, shape=[None, None], name='indices')
        values = tf.placeholder(dtype=tf.float32, shape=[None], name='values')
        shape = tf.placeholder(dtype=tf.int64, shape=[None], name='dense_shape')
        receiver_input = {'indices': indices,
                          'values': values,
                          'dense_shape': shape}
        # 先构成 sparse，然后 sparse_to_dense
        sparse = tf.SparseTensor(indices, values, shape)
        features = {FEATURES_KEY: tf.sparse_tensor_to_dense(sparse)}

        return tf.estimator.export.ServingInputReceiver(features, receiver_input)


    # 在 RoundRobinStrategy 下无法执行
    if ctx.job_name == "chief":
        # 进行 evaluate，比较慢，跳过

        # predictions = cur_e.predict(new_input_fn("test", False))
        # result = cur_e.evaluate(new_input_fn("test", False))
        # with tf.gfile.GFile("{}/evaluate".format(log_dir), 'w') as f:
        #     f.write(str(result))
        #     f.write('\n')
        # 进行预测，分别是测试和训练
        # print('export test result')
        # predictions = estimator.predict(new_input_fn("test", False))
        # print('Writing Predictions to {}'.format(pred_dir))
        # tf.gfile.MakeDirs(pred_dir)
        # with tf.gfile.GFile("{}/test".format(pred_dir), 'w') as f:
        #     for pred in predictions:
        #         f.write(str(pred))
        #         f.write('\n')
        # print('export train result')
        # predictions = estimator.predict(new_input_fn("train", False))
        # print('Writing Predictions to {}'.format(pred_dir))
        # tf.gfile.MakeDirs(pred_dir)
        # with tf.gfile.GFile("{}/train".format(pred_dir), 'w') as f:
        #     for pred in predictions:
        #         f.write(pred['classes'][0])
        #         f.write('\n')
        # 导出模型
        # 191204 这样导出没有办法指定 serving 时的输出，
        cur_e.export_saved_model(export_dir, serving_input_receiver_fn)

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
