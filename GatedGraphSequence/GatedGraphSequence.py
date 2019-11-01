import tensorflow as tf
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D, GRUCell



class GatedGraphSequence(Layer):

    """
    A gated graph sequence layer as presented by Y Li, ICLR16

    Layer based on GraphConv layer from the spektral package.  

 
    This layer computes:
        
        a = tf.matmul(A,h)

        z = self.activation(tf.matmul(a,self.Wz) + tf.matmul(h,self.Uz))
        r = self.activation(tf.matmul(a,self.Wr) + tf.matmul(h,self.Ur))


        h_nh = tf.tanh(tf.matmul(z,self.W) + tf.matmul(tf.multiply(r,h),self.U)) 
        output = tf.multiply((1-z),h) + tf.multiply(z,h_nh)


    where \(h\) is the previous hidden layer output , A is the normalized
    Adjacency matrix, W's and U's are various weight matrices used for the gates and output 

    
    **Input**
    
    - Node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Adj Matrix of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);
    
    **Output**
    
    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.
        
    **Arguments**
    
    - `units`: integer, number of output units;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;  
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    
 
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False


    def build(self, input_shape):

        '''
        Initialize weight matrices. Two for each gate and two for output, as per the original paper. See also Gated Recurrent Unit for more information on gates.

        '''

        self.Wz = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='Wz')
        self.Uz = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='Uz')
    
        self.Wr = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='Wr')
        self.Ur = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='Ur')
        self.W = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='W')
        self.U = self.add_weight(shape=(self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='U')

        return 
    def call(self, inputs):

        h = inputs[0]
        A = inputs[1]

        a = tf.matmul(A,h)


        z = self.activation(tf.matmul(a,self.Wz) + tf.matmul(h,self.Uz))
        r = self.activation(tf.matmul(a,self.Wr) + tf.matmul(h,self.Ur))


        h_nh = tf.tanh(tf.matmul(z,self.W) + tf.matmul(tf.multiply(r,h),self.U)) 
        output = tf.multiply((1-z),h) + tf.multiply(z,h_nh)

        
        return output

    def compute_output_shape(self, input_shape):

        return input_shape 

    def get_config(self):

        config = {
        "units":self.units, 
        "activation":self.activation, 
        "use_bias":self.use_bias, 
        "kernel_initializer":self.kernel_initializer,
        "bias_initializer":self.bias_initializer,
        "kernel_regularizer":self.kernel_regularizer,
        "bias_regularizer":self.bias_regularizer,
        "activity_regularizer":self.activity_regularizer,
        "kernel_constraint":self.kernel_constraint,
        "bias_constraint":self.bias_constraint, 
        "supports_masking":self.supports_masking 

        }
       

        return config


