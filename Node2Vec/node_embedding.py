import numpy as np
import keras
from keras.layers import Layer
import multiprocessing as mp

def get_adjlist(nodes,edges):
    adjlist = []
    for i in nodes:
        adjlist.append([])

    for u,v in edges:
        adjlist[int(u)].append(int(v))
        adjlist[int(v)].append(int(u))
    return adjlist
def get_adjmat(nodes,edges):
    mat = np.zeros((len(nodes),len(nodes)))
    for u,v in edges:
      
        mat[u][v] = 1
        mat[v][u] = 1
    return mat
def isNeighbor(adjlist,u,v):
    return v in adjlist[u]
def getProbs(adjlist, prev,curr,neigs,p,q):
    probs = np.zeros(len(neigs))
    for i in range(len(neigs)):
        if prev == neigs[i]:
            probs[i] = 1/p
        elif isNeighbor(adjlist, prev, neigs[i]):
            probs[i] = 1
        else:
            probs[i] = 1/q
    return probs

def sample_random_walks(graph, num_walks, walk_length, p, q):
    """
    Sampling random walks from a graph. Each walk is a sequence of graph nodes obtained from a random walk.    
    Input: 
        graph: the graph object of networkx.Graph  
        num_walks: the number of random walks 
        walk_length: the length of random walks
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
    Return: 
        walks: an numpy array with shape (num_walks, walk_length)
    """
     ########
            
    walks = []
    nodes = graph.nodes
    edges = graph.edges
    adj_list = get_adjlist(nodes,edges)
    for j in range(num_walks):
        prev = -1
        curr = np.random.randint(len(nodes))

        walk = []
        for i in range(walk_length):
            walk.append(curr)
            
            neigs = adj_list[curr]
            probs = getProbs(adj_list,prev,curr,neigs,p,q)
            normf = sum(probs)
            probs /=normf
            prev = curr
            curr = neigs[np.where(np.random.multinomial(1, probs, size=1)[0] == 1)[0][0]]


        walks.append(np.array(walk))
    walks = np.array(walks)
   
    ########

    return walks 
def walk2pairs(walk, probs, context_size, num_negative_samples, num_nodes):
    #make true pair
    pairs = []
    nodes = list(range(len(probs)))
    for i in range(len(walk)):
        
        bounds = (max(i-context_size,0), min(i+context_size,len(walk)-1))
        a = 0 
        for j in range(bounds[0], bounds[1]+1):
            if not i == j:
                pair = np.array([walk[i],walk[j],1])
                pairs.append(pair)
                a+=1

        
        ######################################
        #make false pairs
        for k in range(a):
            one = walk[i]
            two = np.random.choice(nodes,size = num_negative_samples, p = probs)#np.where(np.random.multinomial(1,probs, size = num_negative_samples)==1)[1] # replace that with np.random.choice
            
            for j in two:
                pair = np.array([one,j,0])
                pairs.append(pair)
    return pairs
    

def collect_skip_gram_pairs(walks, context_size, num_nodes, num_negative_samples, parallel = False):
    """
    Generate positive node pairs from random walks, and also generate random negative pairs 
    Input: 
        walks: numpy.array with shape (num_walks, walk_length). Each row is a random walk with each entry being a graph node 
        context_size: integer, the maximum number of nodes considered before or after the current node
        num_nodes: integer, the size of the original graph generating random walks. (num_nodes - 1) is the largest node index.  
        num_negative_samples: integer, the number of negative nodes to be sampled to get negative pairs.  
    Return: 
        pairs: an numpy array with shape (num_pairs, 3) and type integer. Each row contains a pair of nodes and the label of the pair. 
               If the pair is generated from two nearby nodes in a random walk, then the label is 1. If the pair is generated 
               from a node in the random walk and a random node, then the pair has label 0. The number of pairs is a little less than 
               walks.shape[0] * walks.shape[1] * context_size * 2. In general, each node in `walks` generates `2 * context_size` pairs. 
               But if the node is at the beginning or the end of a walk, it generates less pairs.  
    """
    
    ######################################

    assert(walks.shape[1] >= (2 * context_size + 1))


    # write your code here
    pairs = []
    unique, counts = np.unique(walks, return_counts=True)
    freqs = np.array(list(dict(zip(unique, counts)).values()))
    probs = freqs**3/4
    probs/=sum(probs)
    if not parallel:
        
        for walk in walks:
            tmp = walk2pairs(walk,probs,context_size,num_negative_samples, num_nodes)
            pairs.extend(tmp)

        pairs = np.array(pairs)
        ######################################
        
    else:
        
        
        pool = mp.Pool(mp.cpu_count())
        m = int(len(walks)/mp.cpu_count())
        inds = [m*i for i in range(mp.cpu_count())]
        ps = []
        for i in range(mp.cpu_count()):
            ps.append(pool.apply_async(parallel_collect, args=(walks[inds[i]:inds[i]+m], context_size, num_nodes, num_negative_samples, probs)))
        for i in ps:
            pairs.extend(i.get())
        
        pool.close()
        #pairs = np.array(ps)
   
    return np.array(pairs)

def parallel_collect(walks, context_size, num_nodes, num_negative_samples, probs):
    pairs = []

    for walk in walks:
        tmp = walk2pairs(walk,probs,context_size,num_negative_samples, num_nodes)
        pairs.extend(tmp)

    pairs = np.array(pairs)
    return pairs

class EmbLayer(Layer):
    """
    A keras layer. The layer takes the input of node pairs, looks up embedding vectors for nodes in pairs, and compute 
    the inner product of the vectors in each node pair. 
    """

    def __init__(self, num_nodes=None, emb_dim=None, init_emb=None, output_vecs=None, **kwargs):
        """
        Initialization of the layer. You should provide two ways to initialize the layer. In the first way, provide the shape, 
        (num_nodes, emb_dim), of the embedding matrix. Then the embedding matrix will be initialized internally. In the second 
        way, provide an initial embedding matrix such the embedding matrix will be initialized by the provided matrix. 
        Input: 
            num_nodes: integer, the number of nodes in the graph. Must be provided if init_emb == None. 
            emb_dim: integer, the embedding dimension. Must be provided if init_emb == None. 
            init_emb: numpy.array with size (num_nodes, emb_dim). If this argument is provided, the embedding matrix must be 
                      initialized by this argument, then `num_nodes` or `emb_dim` should have no effect. 
            output_vecs: numpy.array with size (num_nodes, emb_dim). If init_emb is not None, then this argument needs to be 
                         provided, otherwise this argument is neglected.
        Return: 
            
        """
        
        ######################################### 
        
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.init_emb = init_emb
        self.output_vecs = output_vecs
        ######################################### 

        super(EmbLayer, self).__init__(**kwargs) # Be sure to call this at the end 

    def build(self, input_shape):
        """
        Build the keras layer. You should allocate the embedding matrix (also called weights or kernel) as a optimization 
        variable in this function.
        Input:
            input_shape: it should be (num_pairs, 3). It has no use in deciding the shape of the embedding matrix. 
        """

        ######################################### 
        
        if self.num_nodes !=None and self.emb_dim!= None:
            self.kernel = self.add_weight(name = 'kernel', shape = (self.num_nodes,self.emb_dim), initializer = 'uniform', trainable = True)
        else:
           
            
            self.kernel = self.add_weight(name = 'kernel', shape = (self.init_emb.shape[0],self.init_emb.shape[1]), initializer = self.myinit, trainable = True)
      
        ######################################### 

        super(EmbLayer, self).build(input_shape)  # Be sure to call this at the end

   
    def myinit(self, shape, dtype = None):
        a = keras.backend.variable(self.init_emb)
        return a  

    
    def call(self, pairs):
        """
        Here we define the computation (graph) of the layer. Given a pair of nodes, look up the two embedding vectors, 
        take the inner product of the two vectors, and convert it to a probability by the sigmoid function
        Input: 
            pairs: keras tensor with shape (batch_size, 2). Each row is a pair of nodes
        Return: 
            prob: keras tensor with shape (batch_size, ). The probabilities of pair labels being 1 

        """

     
        ####################################
        embds = self.kernel
    
        v = keras.backend.cast(keras.backend.slice(pairs,[0,0],[-1,1]),"int32") 
        u = keras.backend.cast(keras.backend.slice(pairs,[0,1],[-1,1]),"int32") 
        
        vec1 = keras.backend.gather(embds,v)
        vec2 = keras.backend.gather(embds,u)
       
        prod = keras.layers.multiply([vec1,vec2])
        s = keras.backend.sum(prod, axis = 2)
        
        res = keras.backend.sigmoid(s)
        return res

        

    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)

def calcprob(vs):
    u,v = vs
    res = np.dot(u,v)
    res = 1/(1+np.exp(-res))
    return res

def node2vec(graph, num_walks, walk_length, p, q, context_size, num_negative_samples, emb_dim, num_epochs, parallel = False):
    """
    The node2vec algorithm. 
    Input: 
        graph: the graph object of networkx.Graph  
        num_walks: the number of random walks 
        walk_length: the length of random walks
        p: the p parameter. See the node2vec paper 
        q: the q parameter. See the node2vec paper
        context_size: integer, the maximum number of nodes considered before or after the current node
        num_negative_samples: integer, the number of negative nodes to be sampled to get negative pairs.  
        emb_dim: integer, the embedding dimension 
        num_epochs: integer, number of training epochs
    Return: 
        node_emb: an numpy array with shape (num_nodes, emb_dim)
    """
 
    node_emb = np.random.random(size=(graph.number_of_nodes(), emb_dim))

    ############################################################
    # Write your code here
    num_nodes = len(graph.nodes)

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(2,)))
    model.add(EmbLayer(num_nodes=num_nodes, emb_dim=emb_dim))
    model.compile(loss = keras.losses.binary_crossentropy, optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics = ['accuracy'])

    walks = sample_random_walks(graph,num_walks,walk_length, p,q)


    import time

    start = time.time()

    print("parallel = ",parallel)
    pair_labels = collect_skip_gram_pairs(walks, context_size, num_nodes, num_negative_samples, parallel = parallel)
    end = time.time()
    print(end - start)

    node_pairs = pair_labels[:, 0:2]
    labels = pair_labels[:,2]
    print(labels.shape)
    
    model.fit(node_pairs,labels, epochs=num_epochs)    

    node_emb = model.get_weights()[0]

    ############################################################


    return node_emb 

