import copy
import torch as pt
from ...magic import ArgRepr
from ...funcflow import Curry, unit
from ..types import Module, Optimizer, LRScheduler
from .callbacks import TrainCb, EpochCb
from .schedulers import NoSchedule
from .data import TestDataBase, TrainDataBase


class Trainer(ArgRepr):
    """Train and, optionally, evaluate a model with early stopping.

    Parameters
    ----------
    batch_size: int
        Size of the mini-batches to request from the training data.
    max_epochs: int
        Maximum number of epochs to train for.
    loss: Module
        PyTorch module that accepts the output(s) of the model to train as
        first argument(s) and the target as last argument and that produces
        the (scalar) loss to minimize.
    optimizer: Curry[Optimizer]
        A curry of a preconfigured PyTorch Optimizer (or some other custom
        construct) that returns a fully configured PyTorch Optimizer when
        called with the ``model.parameters()`` of the model to train.
    scheduler: Curry[LRScheduler], optional
        A curry of a PyTorch learning-rate scheduler (or some other custom
        construct) that returns a fully configured learning-rate scheduler
        when called with a PyTorch optimizer. Defaults to never changing the
        learning rate at all.
    delay: int, optional
        Number of epochs to wait until the learning rate scheduler is used
        for the first time. Defaults to 0.
    patience: int, optional
        If patience is not ``None`` and smaller than `max_epochs`, early
        stopping is active. A snapshot of the model's state is taken after
        each epoch that improved the loss below its last minimum. If no
        improvement occurs for `patience` epochs, model training is stopped
        (even if `max_epochs` has not been reached yet) and the model is reset
        to its best state. If `delay` > 0, then the early stopping gets active
        only after `delay` epochs have passed.
    max_n: int, optional
        Maximum number of data points to take from the training (and,
        if present, test) data for computing the train (and, optional, test)
        loss after each epoch. Since this is done in a single batch, memory
        consumption might momentarily spike if not set or set too large.
        Defaults to number of data points in test set if present or train set
        if not.
    epoch_cb: EpochCb, optional
        Callback called after each epoch with epoch, train_loss, test_loss,
        and current learning rate. Defaults to no callback.
    train_cb: TrainCb, optional
        Callback called after training finished with last epoch, epoch with the
        best loss, the best loss itself, whether `max_epochs` was exhausted,
        and with the training history in the form of a dictionary of lists
        with train loss, test loss and learning rate.


    Notes
    -----
    Because the (partial) optimizer will simply be completed with
    ``model.parameters()``, parameter groups are not supported. Also, not all
    learning-rate schedulers are supported . Their ``step()`` method is called
    only once after each epoch, and it is called without any arguments.

    """

    def __init__(
            self,
            batch_size: int,
            max_epochs: int,
            loss: Module,
            optimizer: Curry[Optimizer],
            scheduler: Curry[LRScheduler] = Curry[NoSchedule](NoSchedule),
            delay: int = 0,
            patience: int | None = None,
            max_n: int = None,
            epoch_cb: EpochCb = unit,
            train_cb: TrainCb = unit
    ) -> None:
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.delay = delay
        self.patience = max_epochs if patience is None else patience
        self.max_n = max_n
        self.epoch_cb = epoch_cb
        self.train_cb = train_cb
        super().__init__(
            batch_size,
            max_epochs,
            loss,
            max_n,
            optimizer,
            scheduler,
            patience,
            epoch_cb,
        )
        self.history = {'train_loss': [], 'test_loss': [], 'lr': []}

    @property
    def epochs(self) -> range:
        """Iterator over epochs starting with 1 (instead of 0)."""
        return range(1, self.max_epochs + 1)

    def __call__(
            self,
            model: Module,
            train: TrainDataBase,
            test: TestDataBase | None = None
    ) -> Module:
        """Train and, optionally, evaluate a model on train (and test) data.

        Parameters
        ----------
        model: Module
            PyTorch model to train.
        train: TrainDataBase
            Training data.
        test: TestDataBase, optional
            Hold-out data to compute test loss for early stopping (if set up).

        Returns
        -------
        Module
            The trained Pytorch model.

        """

        # Initialize training cycle.
        optimizer = self.optimizer(model.parameters())
        scheduler = self.scheduler(optimizer)

        # How many data points to take for computing train (and test) loss.
        n = train.n if test is None else test.n
        self.max_n = n if self.max_n is None else self.max_n
        max_n = min(self.max_n, n)

        # Initialize counting and accumulation variables.
        epoch = 0
        best_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        n_wait = 1
        max_epochs_reached = False

        # In case we re-use the trainer multiple times, re-initialize history.
        self.history = {'train_loss': [], 'test_loss': [], 'lr': []}

        # Loop over epochs.
        for epoch in self.epochs:
            # Train one epoch, looping over batches.
            model.train()
            for features, targets in train(self.batch_size):
                optimizer.zero_grad()
                predictions = model(*features)
                loss = self.loss(*predictions, targets)
                loss.backward()
                optimizer.step()

            # Evaluate model on train and, potentially, test data.
            model.eval()
            with pt.no_grad():
                features, targets = train.sample(max_n)
                train_loss = self.loss(*model(*features), targets).item()
                if test is not None:
                    features, targets = test.sample(max_n)
                    test_loss = self.loss(*model(*features), targets).item()
                else:
                    test_loss = None

            # Append epoch metrics to training history.
            current_lr = scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            # Call callback with epoch metrics.
            self.epoch_cb(epoch, train_loss, test_loss, current_lr)

            # Update the learning rate of the optimizer if delay is exhausted.
            if epoch >= self.delay:
                scheduler.step()

                # After delaying, check if loss improved within our patience.
                track_loss = train_loss if test_loss is None else test_loss
                if track_loss < best_loss:
                    best_loss = track_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    n_wait = 1
                elif n_wait < self.patience:
                    n_wait += 1
                else:
                    model.load_state_dict(best_state)
                    break

        # Did we exhaust the maximum number of epochs?
        else:
            max_epochs_reached = True

        # Call callback on finished training.
        self.train_cb(
            epoch,
            best_epoch,
            best_loss,
            max_epochs_reached,
            self.history
        )

        # Finally, return the trained model.
        return model
