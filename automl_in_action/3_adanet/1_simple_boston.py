import functools
import os

import adanet
import tensorflow as tf

# 固定种子
RANDOM_SEED = 42

LOG_DIR = 'models'

'''
In this example, we will solve a regression task known as the Boston Housing dataset to 
predict the price of suburban houses in Boston, MA in the 1970s. There are 13 numerical features, 
the labels are in thousands of dollars, and there are only 506 examples.
'''

(x_train, y_train), (x_test, y_test) = (tf.keras.datasets.boston_housing.load_data())
print('Model Inputs: %s \n' % x_train[0])
print('Model Output (house price): $%s' % (y_train[0] * 1000))

FEATURES_KEY = "x"


def input_fn(partition, training, batch_size):
    """Generate an input function for the Estimator."""
    def _input_fn():
        if partition == "train":
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(x_train)
            }, tf.log1p(y_train)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(x_test)
            }, tf.log1p(y_test)))

        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn


_NUM_LAYERS_KEY = "num_layers"


class _SimpleDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet"""

    def __init__(self, optimizer, layer_size, num_layers, learn_mixture_weights, seed):
        """Initializes a `_DNNBuilder`.

        Args:
          optimizer: An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
          layer_size: The number of nodes to output at each hidden layer.
          num_layers: The number of hidden layers.
          learn_mixture_weights: Whether to solve a learning problem to find the
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a no_op
            for the mixture weight train op.
          seed: A random seed.

        Returns:
          An instance of `_SimpleDNNBuilder`.
        """
        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        input_layer = tf.to_float(features[FEATURES_KEY])
        kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        last_layer = input_layer
        for _ in range(self._num_layers):
            last_layer = tf.layers.dense(
                last_layer,
                units=self._layer_size,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer)
        logtis = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=kernel_initializer)

        persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logtis,
            complexity=self._measure_complexity(),
            persisted_tensors=persisted_tensors)

    def _measure_complexity(self):
        """Approximates Rademacher complexity as the square-root of the depth."""
        return tf.sqrt(tf.to_float(self._num_layers))

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
        """See `adanet.subnetwork.Builder`."""
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                       iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""
        if not self._learn_mixture_weights:
            return tf.no_op()
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        if self._num_layers == 0:
            # A DNN with no hidden layers is a linear model
            return "linear"
        return "{}_layer_dnn".format(self._num_layers)


class SimpleDNNGenerator(adanet.subnetwork.Generator):
    """Generates a two DNN subnetworks at each iteration.

    The first DNN has an identical shape to the most recently added subnetwork
    in `previous_ensemble`. The second has the same shape plus one more dense
    layer on top. This is similar to the adaptive network presented in Figure 2 of
    [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
    connections to hidden layers of networks from previous iterations.
    """

    def __init__(self,
                 optimizer,
                 layer_size=64,
                 learn_mixture_weights=False,
                 seed=None):
        """Initializes a DNN `Generator`.

        Args:
          optimizer: An `Optimizer` instance for training both the subnetwork and
            the mixture weights.
          layer_size: Number of nodes in each hidden layer of the subnetwork
            candidates. Note that this parameter is ignored in a DNN with no hidden
            layers.
          learn_mixture_weights: Whether to solve a learning problem to find the
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a no_op
            for the mixture weight train op.
          seed: A random seed.

        Returns:
          An instance of `Generator`.
        """
        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            _SimpleDNNBuilder,
            optimizer=optimizer,
            layer_size=layer_size,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""

        num_layers = 0
        seed = self._seed
        if previous_ensemble:
            num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1].subnetwork.persisted_tensors[_NUM_LAYERS_KEY])
        if seed is not None:
            seed += iteration_number
        return [
            self._dnn_builder_fn(num_layers=num_layers, seed=seed),
            self._dnn_builder_fn(num_layers=num_layers+1, seed=seed)
        ]


"""
Train and evaluate

Next we create an adanet.Estimator using the SimpleDNNGenerator we just defined.

In this section we will show the effects of two hyperparamters: learning mixture weights 
and complexity regularization.

On the righthand side you will be able to play with the hyperparameters of this model. Until you reach 
the end of this section, we ask that you not change them.

At first we will not learn the mixture weights, using their default initial value. Here they will 
be scalars initialized to $1/N$ where $N$ is the number of subnetworks in the ensemble, effectively 
creating a uniform average ensemble.
"""

#@title AdaNet parameters
LEARNING_RATE = 0.001 #@param {type:"number"}
TRAIN_STEPS = 60000 #@param {type:"integer"}
BATCH_SIZE = 32 #@param {type:"integer"}

LEARN_MIXTURE_WEIGHTS = False #@param {type:"boolean"}
ADANET_LAMBDA = 0 #@param {type:"number"}
ADANET_ITERATIONS = 3 #@param {type:"integer"}


def train_and_evaluate(experiment_name, learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
                       adanet_lambda=ADANET_LAMBDA):
    """Trains an `adanet.Estimator` to predict housing prices."""

    model_dir = os.path.join(LOG_DIR, experiment_name)

    estimator = adanet.Estimator(
        # Since we are predicting housing prices, we'll use a regression
        # head that optimizes for MSE.
        head=tf.contrib.estimator.regression_head(
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),

        # Define the generator, which defines our search space of subnetworks
        # to train as candidates to add to the final AdaNet model
        subnetwork_generator=SimpleDNNGenerator(
            optimizer=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE),
            learn_mixture_weights=learn_mixture_weights,
            seed=RANDOM_SEED),

        # Lambda is a the strength of complexity regularization. A larger
        # value will penalize more complex subnetworks.
        adanet_lambda=adanet_lambda,

        # The number of train steps per iteration.
        max_iteration_steps=TRAIN_STEPS // ADANET_ITERATIONS,

        # The evaluator will evaluate the model on the full training set to
        # compute the overall AdaNet loss (train loss + complexity
        # regularization) to select the best candidate to include in the
        # final AdaNet model.
        evaluator=adanet.Evaluator(
            input_fn=input_fn("train", training=False, batch_size=BATCH_SIZE)),

        # Configuration for Estimators
        config=tf.estimator.RunConfig(
            save_summary_steps=5000,
            save_checkpoints_steps=5000,
            tf_random_seed=RANDOM_SEED,
            model_dir=model_dir))

    # Train and evaluate using using the tf.estimator tooling.
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn("train", training=True, batch_size=BATCH_SIZE),
        max_steps=TRAIN_STEPS)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn("test", training=False, batch_size=BATCH_SIZE),
        steps=None,
        start_delay_secs=1,
        throttle_secs=30,
    )
    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def ensemble_architecture(result):
    """Extracts the ensemble architecture from evaluation results."""

    architecture = result["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
    summary_proto = tf.summary.Summary.FromString(architecture)
    return summary_proto.value[0].tensor.string_val[0]


if __name__ == "__main__":
    results, _ = train_and_evaluate("uniform_average_ensemble_baseline")
    print("Loss:", results["average_loss"])
    print("Architecture:", ensemble_architecture(results))