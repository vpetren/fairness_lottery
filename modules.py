import math
import random
from itertools import repeat

import numpy as np
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.prune as pruning

from torch.nn import Parameter
from torch.nn.modules.module import Module


class PruningModulePT(Module):

    def prune_by_percentile(self, amount=0.1, pruning_structure='layerwise', pruning_method='magnitude'):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated.
             Pruning is done either globally or layerwise based.
        Args:
            amount (float): percentile of remaining weights to prune
            pruning_structure (str): How to prune model structurally (globally or layerwise)
            pruning_method (str): Weight pruning criteria (magnitude or random)
        """
        parameters_to_prune = []
        for name, module in self.named_modules():
            if isinstance(module, nn.LSTM):
                names = [name.replace('_orig', '') for name, _ in module.named_parameters()]
                for param_name in names:
                    parameters_to_prune.append((module, param_name))
            elif isinstance(module, nn.Linear):
                names = [name.replace('_orig', '') for name, _ in module.named_parameters()]
                for param_name in names:
                    parameters_to_prune.append((module, param_name))
            elif isinstance(module, nn.Embedding):
                names = [name.replace('_orig', '') for name, _ in module.named_parameters()]
                for param_name in names:
                    parameters_to_prune.append((module, param_name))
            else:
                pass

        if pruning_method == 'magnitude':
            pruning_global = pruning.L1Unstructured
            pruning_local = pruning.l1_unstructured
        elif pruning_method == 'random':
            pruning_global = pruning.RandomUnstructured
            pruning_local = pruning.random_unstructured
        else:
            raise NotImplementedError('Invalid pruning method or not implemented yet.')

        if pruning_structure == 'global':
            pruning.global_unstructured(
                parameters_to_prune,
                pruning_method=pruning_global,
                amount=amount,
            )
        elif pruning_structure == 'layerwise':
            for module, param_name in parameters_to_prune:
                pruning_local(module, name=param_name, amount=amount)
        else:
            raise NotImplementedError('Invalid pruning structure, use layerwise or global')


class DRO_loss(Module):
    def __init__(self, eta, k):
        super(DRO_loss, self).__init__()
        self.eta = eta
        self.k = k
        #self.logsig = torch.nn.LogSigmoid()
        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        loss = self.criterion(x, y)
        #loss = -1 * y * self.logsig(x) - (1 - y) * self.logsig(-x)
        if self.k > 0:
            loss = self.relu(loss - self.eta)
            loss = loss**self.k
            return loss.mean()
        else:
            return loss.mean()
