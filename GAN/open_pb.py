import six
import collections
import logging
import functools
import inspect

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf


if hasattr(inspect, "FullArgSpec"):
    FullArgSpec = inspect.FullArgSpec  # pylint: disable=invalid-name
else:
    _FullArgSpec = collections.namedtuple("FullArgSpec", [
      "args", "varargs", "varkw", "defaults", "kwonlyargs", "kwonlydefaults",
      "annotations"])


def z_generator(shape, distribution_fn=tf.random.uniform, minval=-1.0, maxval=1.0, stddev=1.0, name=None):
    return call_with_accepted_args(distribution_fn, shape=shape, minval=minval, maxval=maxval,
                                   stddev=stddev, name=name)


def call_with_accepted_args(fn, **kwargs):
    """Calls `fn` only with the keyword arguments that `fn` accepts."""
    kwargs = {k: v for k, v in six.iteritems(kwargs) if _has_arg(fn, k)}
    logging.debug("Calling %s with args %s.", fn, kwargs)
    return fn(**kwargs)


def _has_arg(fn, arg_name):
    while isinstance(fn, functools.partial):
        fn = fn.func
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    arg_spec = _getfullargspec(fn)
    if arg_spec.varkw:
        return True
    return arg_name in arg_spec.args or arg_name in arg_spec.kwonlyargs


def _getfullargspec(fn):
    arg_spec_fn = inspect.getfullargspec if six.PY3 else inspect.getargspec
    try:
        arg_spec = arg_spec_fn(fn)
    except TypeError:
        # `fn` might be a callable object.
        arg_spec = arg_spec_fn(fn.__call__)
    if six.PY3:
        assert isinstance(arg_spec, _FullArgSpec)
        return arg_spec
    return _FullArgSpec(
        args=arg_spec.args,
        varargs=arg_spec.varargs,
        varkw=arg_spec.keywords,
        defaults=arg_spec.defaults,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})


def main():
    model_name = 'compare_gan/results_resnet_cifar10/tfhub/40000/'

    np.random.seed(42)
    batch_size = 64
    # TODO: Lage get_dataset()
    num_test_examples = datasets.get_dataset()
    num_batches = int(np.ceil(num_test_examples / batch_size))
    num_averaging_runs = 10

    # This returns the same as gan.as_module_spec() and is a "ModuleSpec" object
    module_spec = hub.load_module_spec(model_name)
    print("got the module_spec")

    with tf.Graph().as_default():
        tf.set_random_seed(42)
        with tf.Session() as sess:
            generator = hub.Module(
                module_spec,
                name="gen_module",
                tags={"gen", "bs{}".format(batch_size)}
            )
            print("Generator inputs: %s", generator.get_input_info_dict())
            z_dim = generator.get_input_info_dict()["z"].get_shape()[1].value
            z = z_generator(shape=[batch_size, z_dim])
            assert "labels" not in generator.get_input_info_dict()
            inputs = dict(z=z)
            print("inputs: {}".format(inputs))
            generated = generator(inputs=inputs, as_dict=True)["generated"]
            print("hey no error!!")
            print("{}".format(module_spec))

            for i in range(num_averaging_runs):
                print("Generating fake dataset %d%d.", i+1, num_averaging_runs)
                # TODO: lage funksjonene EvalDataSample, og sample_fake_dataset som er fra eval_utils
                fake_dset = EvalDataSample(sample_fake_dataset(sess, generated, num_batches))

main()