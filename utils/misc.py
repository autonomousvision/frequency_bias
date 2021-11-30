import sys
import re
import types
import importlib
import yaml
import torch
import imageio
from torchvision.transforms import Pad

from distutils.util import strtobool
from typing import Any, Tuple


# ------------------------------------------------------------------------------------------

# Taken from https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


# ------------------------------------------------------------------------------------------

# Custom functions
def to_easydict(d):
    """ Convert python dict to EasyDict recursively.
    Args:
        d (dict): dictionary to convert

    Returns:
        EasyDict

    """
    d = d.copy()
    for k, v in d.items():
        if isinstance(d[k], dict):
            d[k] = to_easydict(d[k])
    return EasyDict(d)


def to_dict(d):
    """ Convert EasyDict to python dict recursively.
    Args:
        d (EasyDict): dictionary to convert

    Returns:
        dict

    """
    d = d.copy()
    for k, v in d.items():
        if isinstance(d[k], dict):
            d[k] = to_dict(d[k])
    return dict(d)


def save_config(outpath, config):
    config = to_dict(config)
    with open(outpath, 'w') as f:
        yaml.safe_dump(config, f)


def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (str): path to default config
    '''
    # Load configuration from file itself
    if path is not None:
        with open(path, 'r') as f:
            cfg_special = yaml.safe_load(f)
    else:
        cfg_special = dict()

    if default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return to_easydict(cfg)


def update_recursive(dict1, dict2, allow_new=True):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
        allow_new(bool): allow adding new keys

    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            if not allow_new:
                raise RuntimeError(f'New key {k} in dict2 but allow_new=False')
            dict1[k] = {} if isinstance(v, dict) else None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                v = strtobool(v.lower())
            if not isinstance(v, list) and not isinstance(dict1[k], list):
                argtype = type(dict1[k])
                if argtype is not type(None) and v is not None:
                    v = argtype(v)

            if dict1[k] is not None:
                print(f'Changing {k} ---- {dict1[k]} to {v}')

            dict1[k] = v


def args_to_dict(args):
    out = {}
    for k, v in zip(args[::2], args[1::2]):
        assert k.startswith('--'), f'Can only process kwargs starting with "--" but key is {k}'
        k = k.replace('--', '', 1)
        keys = k.split(':')
        Nk = len(keys)
        curr_dct = out
        for i, k_i in enumerate(keys):
            if i == (Nk-1):
                curr_dct[k_i] = v
            else:
                if k_i not in curr_dct:
                    curr_dct[k_i] = {}
                curr_dct = curr_dct[k_i]

    return to_easydict(out)


def count_trainable_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def to_image(timg):
    """Convert tensor image in range [-1, 1] to range [0, 255]."""
    assert timg.dtype == torch.float
    return ((timg / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)


def make_video(images, filenpath, **kwargs):
    # ensure output sizes are even
    H, W = images[0].size
    pad = Pad((0, 0, H%2, W%2), fill=1)
    images = [pad(img) for img in images]
    imageio.mimwrite(filenpath, images, **kwargs)
