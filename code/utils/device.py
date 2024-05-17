import torch as tch


def device(name):
    return tch.device(name if tch.cuda.is_available() else "cpu")
