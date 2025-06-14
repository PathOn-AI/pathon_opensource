import functools
import msgpack
import numpy as np
import torch


def pack_array(obj):
    """Pack numpy arrays and PyTorch tensors for msgpack serialization."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    elif isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    elif isinstance(obj, torch.Tensor):
        cpu_tensor = obj.detach().cpu().numpy()
        return {
            b"__tensor__": True,
            b"data": cpu_tensor.tobytes(),
            b"dtype": cpu_tensor.dtype.str,
            b"shape": cpu_tensor.shape,
            b"device": str(obj.device).encode(),
        }
    return obj


def unpack_array(obj):
    """Unpack numpy arrays and PyTorch tensors from msgpack data."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], 
            dtype=np.dtype(obj[b"dtype"]), 
            shape=obj[b"shape"]
        )
    elif b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    elif b"__tensor__" in obj:
        arr = np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"]
        )
        tensor = torch.from_numpy(arr.copy())
        device = obj[b"device"].decode()
        return tensor.to(device) if device != "cpu" else tensor
    return obj


# Create custom msgpack functions with tensor support
Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)