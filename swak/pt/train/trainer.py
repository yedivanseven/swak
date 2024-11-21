import torch as pt
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
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
    loss: Module
        PyTorch module that accepts the output(s) of the model to train as
        first argument(s) and the target as last argument and that produces
        the (scalar) loss to minimize, i.e., reduction must be "mean" or "sum".
    optimizer: Curry[Optimizer], optional
        A curry of a preconfigured PyTorch Optimizer (or some other custom
        construct) that returns a fully configured PyTorch Optimizer when
        called with the ``model.parameters()`` of the model to train. Defaults
        to ``AdamW`` with its default parameters.
    batch_size: int, optional
        Size of the mini-batches to request from the training data.
        Defaults to 64.
    max_epochs: int, optional
        Maximum number of epochs to train for. Defaults to 100.
    scheduler: Curry[LRScheduler], optional
        A curry of a PyTorch learning-rate scheduler (or some other custom
        construct) that returns a fully configured learning-rate scheduler
        when called with a PyTorch optimizer. Defaults to never changing the
        learning rate at all.
    warmup: int, optional
        In the beginning of training, the learning-rate `scheduler` will be
        stepped after every optimizer step for this many times. Afterward, it
        will only be stepped at the end of each epoch. Defaults to 0, which
        results in no warmup and the learning-rate `scheduler` being stepped
        at the end of the first epoch for the first time.
    patience: int, optional
        If patience is not ``None`` and smaller than `max_epochs`, early
        stopping is active. A snapshot of the model's state is taken after
        each epoch that improved the loss below its last minimum. If no
        improvement occurs for `patience` epochs, model training is stopped
        (even if `max_epochs` has not been reached yet) and the model is reset
        to its best state.
    max_n: int, optional
        Maximum number of data points to take from the training data to
        evaluate the train loss after each epoch. Defaults to number of data
        points in the test set (if present) or the train set (if not).
    step_freq: int, optional
        For how many batches to accumulate gradients before taking an
        optimization step. Defaults to 1, which corresponds to no accumulation.
    clip_grad: float, optional
        Clip gradients such that their overall norm is capped by the given
        value. Defaults to 1.0
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

    Important
    ---------
    optimizer
        Because the (partial) optimizer will simply be completed with
        ``model.parameters()``, parameter groups are not supported.
    scheduler
        During `warmup`, the scheduler is called after each optimizer step and,
        afterward, at the end of each epoch. The scheduler thus needs to be
        aware of the `warmup` settings and act accordingly.
    step_freq
        If the ``reduction`` of the `loss` is "mean", the loss of each batch is
        divided by this number before performing the backward pass. Strictly
        speaking, each batch should thus have the exact same number of data
        points in this case. Furthermore, complications might arise when
        the model to train contains elements like ``BatchNorm``.

    See Also
    --------
    InMemory
    EpochCallback
    EpochPrinter
    TrainPrinter

    """

    def __init__(
            self,
            loss: Module,
            optimizer: Curry[Optimizer] = Curry[AdamW](AdamW),
            batch_size: int = 64,
            max_epochs: int = 100,
            scheduler: Curry[LRScheduler] = Curry[NoSchedule](NoSchedule),
            warmup: int = 0,
            patience: int | None = None,
            max_n: int | None = None,
            step_freq: int = 1,
            clip_grad: float = 1.0,
            checkpoint: Checkpoint = InMemory(),
            epoch_cb: EpochCallback = EpochPrinter(),
            train_cb: TrainCallback = TrainPrinter()
    ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.warmup = warmup
        self.patience = max_epochs if patience is None else patience
        self.max_n = max_n
        self.step_freq = step_freq
        self.clip_grad = clip_grad
        self.checkpoint = checkpoint
        self.epoch_cb = epoch_cb
        self.train_cb = train_cb
        super().__init__(
            loss,
            optimizer,
            batch_size,
            max_epochs,
            scheduler,
            warmup,
            patience,
            max_n,
            checkpoint,
            epoch_cb,
            train_cb
        )
        self.history = {'train_loss': [], 'test_loss': [], 'lr': []}

    @property
    def scale(self) -> float:
        """Scaling factor for the accumulated loss before the backward pass."""
        return 1.0 / self.step_freq if self.loss.reduction == 'mean' else 1.0

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
            # Prepare the model for training
            model.train()
            model.zero_grad(set_to_none=True)
            # Get an iterator over batches for one epoch of training data.
            n_batches, batches = train(self.batch_size, self.step_freq, epoch)
            # Initialize a progress bar to monitor training in real time.
            progress = tqdm(batches, 'Batches', n_batches, False)
            # Loop over batches for one epoch of training data
            for batch_index, (features, target) in enumerate(progress, 1):
                loss = self.loss(*model(*features), target)
                # Scale gradients for multi-batch accumulation if required.
                (self.scale * loss).backward()
                # Clip gradients if greater than specified maximum norm.
                norm = clip_grad_norm_(model.parameters(), self.clip_grad)
                # Report loss and norm to the tqdm progress bar for feedback.
                progress.set_postfix(loss=f'{loss:4.2f}', grad=f'{norm:4.2f}')
                # Step after accumulating gradients for step_freq batches.
                if batch_index % self.step_freq == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # During warmup, step the scheduler after each opti step.
                    if scheduler.last_epoch < self.warmup:
                        scheduler.step()

            # How many data points to take for computing train (and test) loss.
            n = train.n if test is None else test.n
            max_n = n if self.max_n is None else min(self.max_n, n)

            # Evaluate model on training data ...
            n = 0
            train_loss = 0.0
            model.eval()
            with pt.no_grad():
                for features, target in train.sample(self.batch_size, max_n):
                    loss = self.loss(*model(*features), target).item()
                    if self.loss.reduction == 'mean':
                        n_new = target.shape[0]
                        train_loss += n_new * (loss - train_loss) / (n + n_new)
                        n += n_new
                    else:
                        train_loss += loss

            # ... and, if present, on test data.
            if test is None:
                test_loss = float('nan')
            else:
                n = 0
                test_loss = 0.0
                with pt.no_grad():
                    for features, target in test.sample(self.batch_size):
                        loss = self.loss(*model(*features), target).item()
                        if self.loss.reduction == 'mean':
                            n_new = target.shape[0]
                            test_loss += n_new * (loss - test_loss)/(n + n_new)
                            n += n_new
                        else:
                            test_loss += loss

            # Append epoch metrics to training history.
            current_lr = scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            # Call callback with epoch metrics and information.
            if test is None:
                sample = train.sample(self.batch_size, max_n)
            else:
                sample = test.sample(self.batch_size)
            self.epoch_cb(
                epoch,
                train_loss,
                test_loss,
                current_lr,
                model,
                sample
            )

            # After warmup, step the scheduler at the end of the epoch.
            if scheduler.last_epoch >= self.warmup:
                scheduler.step()

            # Check if the loss improved within our patience.
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
