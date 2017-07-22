import numpy as np
import theano
import theano.tensor as T
#import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        self.layer_inputs = layer_input
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # The current list of input nodes
        self.layer_inputs = None
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = 0.0
        #self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
        #                                               shape=[],
        #                                               name='use_dropout')

        # Dictionary to store variables by names
        self.var_dict = {}
        # List to store variables needed to compute gradients
        self.grad_params = []
        # List to store update pairs
        self.update_params = []

        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding="latin1").item()
        for op_name in data_dict:
            for param_name, data in data_dict[op_name].items():
                try:
                    name = op_name + param_name
                    shared_var = self.var_dict[name]
                    shared_var.set_value(data)
                except ValueError:
                    if not ignore_missing:
                        raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_grad_params(self):
        '''Returns the params to be updated by gradients'''
        return self.grad_params

    def get_update_params(self):
        '''Returns the params to be updated by pairs'''
        return self.update_params

    def get_output_second2last(self):
        '''Return the output of the second to the last layer'''
        return self.layer_inputs

    def get_layer_output(self, layer_name):
        '''Return the output of specified layer by layer name'''
        return self.layers[layer_name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in list(self.layers.items())) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             pad_h=0,
             pad_w=0,
             group=1,
             biased=True):

        convolve = lambda i, k: T.nnet.conv2d(i, k, border_mode=(pad_h, pad_w),
                                              subsample=(s_h, s_w),
                                              filter_flip=False)
        kernel_data = np.ones((c_o, 1, k_h, k_w), dtype=np.float32)
        kernel = theano.shared(kernel_data)
        self.var_dict[name + "weights"] = kernel
        self.grad_params.append(kernel)

        if group == 1:
            output = convolve(input, kernel)
        else:
            c_i = T.shape(input)[1]
            input_groups = T.split(input, [c_i // group] * group, group, 1)
            kernel_groups = T.split(kernel, [c_o // group] * group, group, 0)
            output_groups = [convolve(i, k) for i, k in zip(input_groups,
                                                            kernel_groups)]
            output = T.concatenate(output_groups, 1)

        if biased:
            biased_data = np.ones(c_o, dtype=np.float32)
            biases = theano.shared(biased_data)
            self.grad_params.append(biases)
            self.var_dict[name + "biases"] = biases
            output = output + biases.dimshuffle('x', 0, 'x', 'x')

        if relu:
            output = T.nnet.relu(output)
        return output

    @layer
    def relu(self, input, name):
        return T.nnet.relu(input)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, pad_h=0, pad_w=0):
        return T.signal.pool.pool_2d(input, ws=(k_h, k_w), stride=(s_h, s_w),
                                     pad=(pad_h, pad_w), mode='max',
                                     ignore_border=True)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, pad_h=0, pad_w=0):
        return T.signal.pool.pool_2d(input, ws=(k_h, k_w), stride=(s_h, s_w),
                                     pad=(pad_h, pad_w),
                                     mode='average_inc_pad',
                                     ignore_border=True)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        input_sqr = T.sqr(input)
        b, ch, r, c = T.shape(input)
        extra_channels = T.alloc(0., b, ch + 2 * radius, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, radius:radius+ch, :, :],
                                    input_sqr)
        scale = bias
        for i in range(radius * 2 + 1):
            scale += alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** beta
        return input / scale

    @layer
    def concat(self, inputs, axis, name):
        return T.concatenate(inputs, 1)

    @layer
    def add(self, inputs, name):
        output = None
        for i in inputs:
            if output is None:
                output = i
            else:
                output = output + i
        return output

    @layer
    def fc(self, input, num_out, name, relu=True):
        weights_name = name + "weights"
        bias_name = name + "biases"
        if input.ndim == 4:
            input = input.reshape(shape=(-1, T.prod(input.shape[1:])),
                                  ndim=2)

        weights_data = np.ones((1, 1), dtype=np.float32)
        weights = theano.shared(weights_data)
        self.var_dict[weights_name] = weights
        self.grad_params.append(weights)

        biases_data = np.ones(num_out, dtype=np.float32)
        biases = theano.shared(biases_data)
        self.var_dict[bias_name] = biases
        self.grad_params.append(biases)

        output = T.dot(input, weights) + biases
        if relu:
            output = T.nnet.relu(output)
        return output

    @layer
    def softmax(self, input, name):
        return T.nnet.softmax(input)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        scale_data = np.ones(1, dtype=np.float32)
        scale_name = name + 'scale'
        scale = theano.shared(scale_data)
        self.var_dict[scale_name] = scale
        self.grad_params.append(scale)

        offset_data = np.ones(1, dtype=np.float32)
        offset_name = name + 'offset'
        offset = theano.shared(offset_data)
        self.var_dict[offset_name] = offset
        self.grad_params.append(offset)

        mean_data = np.ones(1, dtype=np.float32)
        mean_name = name + 'mean'
        mean = theano.shared(mean_data)
        self.var_dict[mean_name] = mean

        variance_data = np.ones(1, dtype=np.float32)
        variance_name = name + 'variance'
        variance = theano.shared(variance_data)
        self.var_dict[variance_name] = variance

        if self.trainable:
            output, outmean, invstd, new_running_mean, new_running_var = (
                T.nnet.bn.batch_normalization_train(input, gamma=scale,
                    beta=offset, axes='spatial', running_mean=mean,
                    running_var=variance)
            )

            self.update_params.append((mean, new_running_mean))
            self.update_params.append((variance, new_running_var))
        else:
            output = T.nnet.bn.batch_normalization_test(input, gamma=scale,
                        beta=offset, mean=mean, var=variance, axes='spatial')

        if relu:
            output = T.nnet.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        return input
        #keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        #return tf.nn.dropout(input, keep, name=name)
