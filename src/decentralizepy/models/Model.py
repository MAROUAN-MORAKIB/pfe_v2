import pickle
from pathlib import Path

import torch
from torch import nn


class Model(nn.Module):
    """
    This class wraps the torch model
    More fields can be added here

    """

    def __init__(self):
        """
        Constructor

        """
        super().__init__()
        self.model_change = None
        self._param_count_ot = None
        self._param_count_total = None
        self.accumulated_changes = None
        self.shared_parameters_counter = None

    def count_params(self, only_trainable=False):
        """
        Counts the total number of params

        Parameters
        ----------
        only_trainable : bool
            Counts only parameters with gradients when True

        Returns
        -------
        int
            Total number of parameters

        """
        if only_trainable:
            if not self._param_count_ot:
                self._param_count_ot = sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                )
            return self._param_count_ot
        else:
            if not self._param_count_total:
                self._param_count_total = sum(p.numel() for p in self.parameters())
            return self._param_count_total

    def rewind_accumulation(self, indices):
        """
        resets accumulated_changes at the given indices

        Parameters
        ----------
        indices : torch.Tensor
            Tensor that contains indices corresponding to the flatten model

        """
        if self.accumulated_changes is not None:
            self.accumulated_changes[indices] = 0.0

    def dump_weights(self, directory, uid, round):
        # Ensure the directory exists
        
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), Path(directory) / f"{round}_weight_{uid}.pt")
        


    def get_weights(self):
        """
        flattens the current weights

        """
        with torch.no_grad():
            tensors_to_cat = []
            for _, v in self.state_dict().items():
                tensors_to_cat.append(v.flatten())
            flat = torch.cat(tensors_to_cat)

        return flat
