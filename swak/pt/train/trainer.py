import torch as pt
from tqdm import tqdm
from ...misc import ArgRepr
from ...funcflow import Curry
from ..types import Module, Optimizer, LRScheduler
from ..exceptions import TrainError
from .callbacks import EpochCallback, EpochPrinter, TrainCallback, TrainPrinter
from .checkpoints import Checkpoint, InMemory
from .schedulers import NoSchedule
from .data import TestDataBase, TrainDataBase

__all__ = ['Trainer']


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
    warmup: int, optional
        Number of epochs to wait until the learning rate scheduler is used
        for the first time. Defaults to 0.
    patience: int, optional
        If patience is not ``None`` and smaller than `max_epochs`, early
        stopping is active. A snapshot of the model's state is taken after
        each epoch that improved the loss below its last minimum. If no
        improvement occurs for `patience` epochs, model training is stopped
        (even if `max_epochs` has not been reached yet) and the model is reset
        to its best state. If `warmup` > 0, then the early stopping gets active
        only after `warmup` epochs have passed.
    max_n: int, optional
        Maximum number of data points to take from the training (and,
        if present, test) data for computing the train (and, optional, test)
        loss after each epoch. Since this is done in a single batch, memory
        consumption might momentarily spike if not set or set too large.
        Defaults to number of data points in test set if present or train set
        if not.
    checkpoint: Checkpoint, optional
        Whenever the train (or test) loss after an epoch is smaller than the
        loss after the last, a new snapshot of the model state is saved by
        calling the `save` method of the `checkpoint` instance. Defaults to
        ``InMemory``.
    epoch_cb: EpochCallback, optional
        Callback called after each epoch with epoch, train_loss, test_loss,
        and current learning rate. Defaults to ``EpochPrinter``.
    train_cb: TrainCallback, optional
        Callback called after training finished with last epoch, epoch with the
        best loss, the best loss itself, whether `max_epochs` was exhausted,
        and with the training history in the form of a dictionary of lists
        with train loss, test loss and learning rate.
        Defaults to ``TrainPrinter``.

    Warnings
    --------
    optimizer
        Because the (partial) optimizer will simply be completed with
        ``model.parameters()``, parameter groups are not supported.
    scheduler
        Not all learning-rate schedulers are supported . Their ``step()``
        method is called only once after each epoch, and it is called without
        any arguments.

    See Also
    --------
    InMemory
    EpochCallback
    EpochPrinter
    TrainPrinter

    """

    def __init__(
            self,
            batch_size: int,
            max_epochs: int,
            loss: Module,
            optimizer: Curry[Optimizer],
            scheduler: Curry[LRScheduler] = Curry[NoSchedule](NoSchedule),
            warmup: int = 0,
            patience: int | None = None,
            max_n: int = None,
            checkpoint: Checkpoint = InMemory(),
            epoch_cb: EpochCallback = EpochPrinter(),
            train_cb: TrainCallback = TrainPrinter()
    ) -> None:
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup = warmup
        self.patience = max_epochs if patience is None else patience
        self.max_n = max_n
        self.checkpoint = checkpoint
        self.epoch_cb = epoch_cb
        self.train_cb = train_cb
        super().__init__(
            batch_size,
            max_epochs,
            loss,
            optimizer,
            scheduler,
            warmup,
            patience,
            max_n,
            checkpoint,
            epoch_cb,
            train_cb
        )
        self.history = {'train_loss': [], 'test_loss': [], 'lr': []}

    def train(
            self,
            model: Module,
            train: TrainDataBase,
            test: TestDataBase | None = None
    ) -> Module:
        """Train a fresh model from scratch, starting from a clean slate.

        Parameters
        ----------
        model: Module
            PyTorch model to train. Must have a ``reset_parameters()`` method
            that can be called without any parameters to re-initialize all
            trainable model parameters and buffers.
        train: TrainDataBase
            Training data.
        test: TestDataBase, optional
            Hold-out data to compute test loss. If configured, early stopping
            will track the `test` loss if `test` is given and the `train` loss
            if it is not.

        Returns
        -------
        Module
            The trained Pytorch model.

        Raises
        ------
        TrainError
            If the `model` does not have a ``reset_parameters()`` method.

        Important
        ---------
        The `model` to train must always return a *tuple* of tensors, not just
        a single tensor. If it produces only one tensor, return it as a 1-tuple
        of tensors.

        Warnings
        --------
        Model training is (re-)started from scratch every time this method is
        called. The training history is erased, all internal model parameters
        are reset to a pristine initial state, and previously saved checkpoints
        are irrevocably deleted!

        """
        self.checkpoint.reset_parameters()
        self.history = {'train_loss': [], 'test_loss': [], 'lr': []}
        try:
            model.reset_parameters()
        except AttributeError as error:
            msg = 'Models must have a "reset_parameters()" method!"'
            raise TrainError(msg) from error
        return self.resume(model, train, test)

    def resume(
            self,
            model: Module,
            train: TrainDataBase,
            test: TestDataBase | None = None
    ) -> Module:
        """Resume model training from the best epoch checkpointed so far.

        Parameters
        ----------
        model: Module
            PyTorch model to train.
        train: TrainDataBase
            Training data.
        test: TestDataBase, optional
            Hold-out data to compute test loss. If configured, early stopping
            will track the `test` loss if `test` is given and the `train` loss
            if it is not.

        Returns
        -------
        Module
            The trained Pytorch model.

        Note
        ----
        This is safe to use even if you have never trained your model before,
        and you are starting from scratch.

        Important
        ---------
        The `model` to train must always return a *tuple* of tensors, not just
        a single tensor. If it produces only one tensor, return it as a 1-tuple
        of tensors.

        """
        # How many data points to take for computing train (and test) loss.
        n = train.n if test is None else test.n
        self.max_n = n if self.max_n is None else self.max_n
        max_n = min(self.max_n, n)

        # How many batches do we have?
        n_batches = train.n // self.batch_size + 1

        # Initialize training cycle.
        optimizer = self.optimizer(model.parameters())
        scheduler = self.scheduler(optimizer)
        epoch, best_loss = self.checkpoint.load(model, optimizer, scheduler)

        # Initialize counting and accumulation variables.
        best_epoch = epoch
        n_wait = 1
        max_epochs_reached = False

        # Loop over epochs.
        for epoch in range(epoch + 1, self.max_epochs + 1):
            # Train one epoch, looping over batches.
            model.train()
            data = tqdm(train(self.batch_size), 'Batches', n_batches)
            for features, target in data:
                optimizer.zero_grad()
                predictions = model(*features)
                loss = self.loss(*predictions, target)
                loss.backward()
                optimizer.step()

            # Evaluate model on train and, potentially, test data.
            model.eval()
            with pt.no_grad():
                features, target = train.sample(max_n)
                train_loss = self.loss(*model(*features), target).item()
                if test is None:
                    test_loss = float('nan')
                else:
                    features, target = test.sample(max_n)
                    test_loss = self.loss(*model(*features), target).item()

            # Append epoch metrics to training history.
            current_lr = scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            # Call callback with epoch metrics and information.
            self.epoch_cb(
                epoch,
                train_loss,
                test_loss,
                current_lr,
                model,
                features,
                target
            )

            # Update the learning rate of the optimizer if warmup is exhausted.
            if epoch >= self.warmup:
                scheduler.step()

                # After warm-up, check if loss improved within our patience.
                track_loss = train_loss if test is None else test_loss
                if track_loss < best_loss:
                    best_loss = track_loss
                    best_epoch = epoch
                    n_wait = 1
                    self.checkpoint.save(
                        best_epoch,
                        best_loss,
                        model,
                        optimizer,
                        scheduler
                    )
                elif n_wait < self.patience:
                    n_wait += 1
                else:
                    self.checkpoint.load(model, optimizer, scheduler)
                    break

        # Did we exhaust the maximum number of epochs?
        else:
            max_epochs_reached = True

        # Call callbacks on finished training.
        self.epoch_cb.close()
        self.train_cb(
            epoch,
            best_epoch,
            best_loss,
            max_epochs_reached,
            self.history
        )

        # Finally, return the trained model.
        return model
