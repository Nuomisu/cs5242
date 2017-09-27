import numpy as np


def sgd(w, dw, learning_rate=None):
    ## Stochastic gradient descent
    if learning_rate is None: learning_rate = 1e-2
  
    w -= learning_rate * dw
    return w, learning_rate

def forward(x, w, b):
    af, af_cache = forward_affine(x, w, b)
    out, relu_cache = forward_relu(af)
    cache = (af_cache, relu_cache)
    return out, cache

def forward_affine(x, w, b):
    result = np.dot(x, w) + b
    cache = (x, w, b)
    return result, cache

def forward_relu(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def backward(out, cache):
    af_cache, relu_cache = cache
    d_af = backward_relu(out, relu_cache)
    d_x, d_w, d_b = backward_affine(d_af, af_cache)
    return d_x, d_w, d_b

def backward_affine(out, cache):
    x, w, b = cache
    d_x, d_w, d_b = None, None, None
    d_w = np.dot(x.T, out)
    d_b = np.sum(out, axis = 0)
    d_x = np.dot(out, w.T)
    return d_x, d_w, d_b


def backward_relu(out, cache):
    x = cache
    d_x = out 
    d_x[cache <= 0] = 0
    return d_x


def cross_entropy(x, y):
    x= x.astype(np.float32)
    probs = np.exp(x)
    #probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    logprobs = []
    for i in xrange(probs.shape[0]):
        logprobs.append(probs[i][y[i]])
    corect_logprobs = -np.log(logprobs)
    loss = np.sum(corect_logprobs)
    loss = loss / N
    #loss = 0
    dx = probs.copy()
    for i in xrange(probs.shape[0]):
        dx[i][y[i]] -= 1
    dx /= N
    return loss, dx


def adam(x, dx, config=None):
    if config is None: 
        config = {}
        config['learning_rate'] = 1e-3
        config['beta1'] = 0.9
        config['beta2'] = 0.999
        config['epsilon'] = 1e-8
        config['m'] = np.zeros_like((x))
        config['v'] = np.zeros_like((x))
        config['t'] = 0
    next_x = None
    learning_rate, beta1, beta2, eps, m, v, t \
        = config['learning_rate'], config['beta1'], config['beta2'], \
        config['epsilon'], config['m'], config['v'], config['t']
    t += 1
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx**2)
    # bias correction:
    mb = m / (1 - beta1**t)
    vb = v / (1 - beta2**t)
    next_x = -learning_rate * mb / (np.sqrt(vb) + eps) + x
    config['m'], config['v'], config['t'] = m, v, t
    return next_x, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
        config['learning_rate'] = 1e-1
        config['momentum'] = 0.6
        config['velocity'] = np.zeros_like(w)
    v = config['velocity']
    next_v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + next_v
    config['velocity'] = next_v
    return next_w, config

def rmsprop(x, dx, config=None):
    if config is None:
        config = {}
        config['learning_rate'] = 1e-1
        config['decay_rate'] = 0.99
        config['epsilon'] = 1e-8
        config['cache'] = np.zeros_like(x)
    config['cache'] = config['cache'] * config['decay_rate'] +\
        (1 - config['decay_rate']) * dx**2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']
                                                         + config['epsilon']))
    return next_x, config
