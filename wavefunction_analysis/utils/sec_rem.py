import numpy as np

import warnings

def put_keys_kwargs_to_object(obj, key={}, **kwargs):
    """
    put the individual keys in the dictionary first
    then use dictionary to assign the obj class attributes
    """
    keys = put_kwargs_to_keys(key, **kwargs)
    put_kwargs_to_object(obj, **keys)


def put_kwargs_to_keys(key={}, **kwargs):
    """
    put all the individual keyword and value into the key dictionary
    """
    for name, value in kwargs.items():
        if name in key.keys():
            warnings.warn('keyword %s is over written as %s' % (name, value), DeprecationWarning)
        key[name] = value

    return key


def put_keys_to_kwargs(key, **kwargs):
    """
    put the values in key dict to kwargs dict
    """
    for name, value in key.items():
        if name in  kwargs.keys():
            warnings.warn('keyword %s is over written as %s' % (name, value), DeprecationWarning)
        kwargs[name] = value

    return kwargs


def put_kwargs_to_object(obj, **kwargs):
    """
    put all the keywords and values into the obj class
    """
    class_variables = kwargs.pop('class_variables', None)
    if class_variables:
        for var in class_variables:
            for name, value in kwargs.items():
                if name == var:
                    setattr(obj, name, value)

    else:
        for name, value in kwargs.items(): # put all the variables in the class
            if isinstance(value, list):
                value = np.array(value)
            setattr(obj, name, value)
        #else:
        #    raise Exception('%s class does not have %s attribute' % (obj.__class__.__name__, name))


def put_keys_to_object(obj, key):
    """
    put all the keywords and values into the obj class
    """
    put_kwargs_to_object(obj, **key)
