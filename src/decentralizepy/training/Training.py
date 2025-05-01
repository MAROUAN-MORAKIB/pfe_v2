import logging

import torch
import json
import os
from decentralizepy import utils


class Training:
    """
    This class implements the training module for a single node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def reset_optimizer(self, optimizer):
        """
        Replace the current optimizer with a new one

        Parameters
        ----------
        optimizer : torch.optim
            A new optimizer

        """
        self.optimizer = optimizer

    def eval_loss(self, dataset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                data = data.to(self.device)
                target = target.to(self.device)
                # output = self.model(data)
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def trainstep(self, data, target):
        """
        One training step on a minibatch.

        Parameters
        ----------
        data : any
            Data item
        target : any
            Label

        Returns
        -------
        int
            Loss Value for the step

        """
        data = data.to(self.device)
        target = target.to(self.device)
        self.model.zero_grad()
        output = self.model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        for epoch in range(self.rounds):
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                logging.debug(
                    "Starting minibatch {} with num_samples: {}".format(
                        count, len(data)
                    )
                )
                logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))



    def train(self, dataset):
        self.model.train()
        iter_loss = 0.0
        count = 0
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)

        loss_values = []  # ✅ Store losses per node

        while count < self.rounds:
            for data, target in trainset:
                iter_loss += self.trainstep(data, target)
                count += 1
                avg_loss = iter_loss / count
                logging.info(f"Node {self.rank} - Round {count} loss: {avg_loss}")
                loss_values.append(avg_loss)  # ✅ Append loss at each step
                
                if count >= self.rounds:
                    break
        
        # ✅ Ensure the directory exists
        os.makedirs("logs/nodes", exist_ok=True)

        # ✅ Append loss values instead of overwriting
        filename = f"logs/nodes/train_loss_node_{self.rank}.json"
        
        if os.path.exists(filename):
            with open(filename, "r") as f:
                existing_loss = json.load(f)
        else:
            existing_loss = []

        # ✅ Combine old and new losses
        total_loss_values = existing_loss + loss_values

        # ✅ Save updated loss history
        with open(filename, "w") as f:
            json.dump(total_loss_values, f)

        logging.info(f"Node {self.rank} training loss saved to {filename}")

