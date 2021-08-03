import numpy as np
import torch
import copy
import functools


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def original_initialization(model, state_dict):
    for name, p in model.named_parameters():
        if "embedding" in name:
            p.data = model.state_dict()[name]
            continue
        if "weight" in name or "weight_prune" in name:
            # print("a", name)
            m = name.split(".")[0]
            p.data = model.state_dict()[f"{m}.mask"] * state_dict[name]
        if "bias" in name:
            # print("b", name)
            p.data = model.state_dict()[name]

def print_model_sparsity(model):
    glob_nz_count, glob_n = 0, 0

    for name, param in model.named_parameters():
        name = name.replace('_orig', '')
        weights = rgetattr(model, name)
        n_params = int(weights.data.nelement())
        n_zeros = int(torch.sum(weights.data == 0))
        nz_count = n_params - n_zeros
        glob_n += n_params
        glob_nz_count += nz_count
        print(f'{name:25} | non_zeros = {nz_count:7} / {n_params:7} ({100 * nz_count / n_params:6.2f}%) | total_pruned = {n_params - nz_count :7} | shape = {param.shape}')

    print(f'alive: {glob_nz_count}, pruned : {glob_n - glob_nz_count}, total: {glob_n}, Compression rate : {glob_n/glob_nz_count:10.2f}x  ({100 * (glob_n-glob_nz_count) / glob_n:6.2f}% pruned)')


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def reset_initialization(model, state_dict):
    state_copy = copy.deepcopy(state_dict)
    for name, p in model.named_parameters():

        _name = name.replace('_orig', '')
        mask = rgetattr(model, f"{_name}_mask")

        rsetattr(model, f"{name}.data", state_copy[_name])
        rsetattr(model, f"{_name}.data", state_copy[_name] * mask)
