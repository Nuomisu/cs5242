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
    probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    logprobs = []
    for i in xrange(probs.shape[0]):
        logprobs.append(probs[i][y[i]])
    corect_logprobs = -np.log(logprobs)
    loss = np.sum(corect_logprobs)
    loss = loss / N
    dx = probs.copy()
    for i in xrange(probs.shape[0]):
        dx[i][y[i]] -= 1
    dx /= N
    return loss, dx


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)
    
    next_x = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config.                                                         #
    #############################################################################
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
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return next_x, config
