import jax.nn as nn
import jax.numpy as jnp
import optax

def cross_entropy_loss(logits, labels, num_classes=10):
    """
    Convert the labels to logits and computes the cross entropy loss between
    the logits and the labels averaged over a batch.

    Cross-Entropy Loss = -sum(labels * log(softmax(logits)))

    Args:
        logits (jnp.array[float]): Tensor of size (batch_size, num_classes) with logits 
        labels (jn.array[int]): Tensor of shape (batch_size, num_classes) with ints representing true classes
    """
    labels_onehot = nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def init_params(model, input_shape, rng):
    """
    Initialize the parameters of the network.
    """
    x = jnp.ones(input_shape)
    params = model.init(rng, x)
    return params