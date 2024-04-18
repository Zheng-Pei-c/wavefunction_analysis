import numpy as np

def put_kwargs_keys_to_object(obj, key={}, **kwargs):
    """
    put the individual keys in the dictionary first
    then use dictionary to assign the obj class attributes
    """
    key = put_kwargs_to_keys(key, **kwargs)
    put_keys_to_object(obj, key)


def put_kwargs_to_keys(key={}, **kwargs):
    """
    put all the individual keyword and value into the key dictionary
    """
    for name, value in kwargs.items():
        if name in key.keys():
            warnings.warn('keyword %s is over written as %s' % (name, value), DeprecationWarning)
        key[name] = value

    return key


def put_keys_to_object(obj, key):
    """
    put all the keywords and values into the obj class
    """
    for name, value in key.items(): # put all the variables in the class
        if hasattr(obj, name):
            if type(value) is list:
                value = np.array(value)
            setattr(obj, name, value)
        #else:
        #    raise Exception('%s class does not have %s attribute' % (obj.__class__.__name__, name))
