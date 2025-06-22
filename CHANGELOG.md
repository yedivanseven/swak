# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2025-06-22
- Added Fallback to funcflow
- Homogenized funcflow behavior on functions returning one-tuples
- Added missing `__hash__` magic methods
- Minor fixes

## [0.5.2] - 2025-06-12
- Fixed JSON logging issue with JsonObject
- Added or-operator ("|") merging to JsonObject

## [0.5.1] - 2025-04-29
- Added log method to functional PassThrough loggers

## [0.5.0] - 2025-04-09
- Added JSON logger

## [0.4.9] - 2025-03-22
- Fixed PyTorch training loop regression

## [0.4.8] - 2025-03-18
- Circumvented pandas warnings

## [0.4.7] - 2025-03-18
- Made pandas partial polymorphous

## [0.4.6] - 2025-03-18
- Added pandas dataframe SortValues, Drop, DropNA, SetIndex, ResetIndex, and Rename partials
- Added pandas group-by RollingGroupBy, RollingGroupByAgg, and RollingGroupByApply partials
- Added first batch of polars dataframe partials
- Added polars GroupByAgg partial
- Added polars FromPandas partial

## [0.4.5] - 2025-03-12
- Changed S3 client caching for AWS tools

## [0.4.4] - 2025-03-11
- Added S3 parquet file to dataframe and the other way around 
- Added polars dependency

## [0.4.3] - 2025-03-07
- Added pandas dataframe assign wrapper

## [0.4.2] - 2025-03-06
- Added pandas dataframe join wrapper

## [0.4.1] - 2025-03-06
- Reinforced consistency in model dimension specification
- Pandas column(s) selector now also accept group-by objects
- Added pandas groupby and groupby.agg wrappers 

## [0.4.0] - 2025-02-10
- Fixed default separator in callback after every batch of model training

## [0.3.9] - 2025-02-10
- Callback after every batch in model training now also get gradient norm
- Removed redundant argument to the YamlWriter
- Added JsonWriter and JsonReader

## [0.3.8] - 2025-01-28
- Added a shorter log format
- Parent directory of log files will now be created if it does not exist
- PyTorch model training loop can now call multiple callbacks

## [0.3.7] - 2025-01-24
- Renamed StdOutLogger to StdLogger and added option to log to stderr
- Renamed PassThroughStdOutLogger to PassThroughStdLogger
- Added option to log to stderr
- Added "close" method to batch callback to stay compatible with TensorBoard
- "close" method of batch callback is now called at the end of model training

## [0.3.6] - 2025-01-22
- Added unit tests to TrainData helper methods.
- Added option to enable/disable progress in PyTorch model training loop
- Added abstract base class for TrainCallbacks
- Added abstract base class and simple printer for StepCallbacks
- Added a callback to be called after each batch to the training loop

## [0.3.5] - 2025-01-06
- Exponential smoothing per-batch loss also for validation errors
- Reset learnable parameters of activation functions
- Added more neural-network building blocks

## [0.3.4] - 2025-01-03
- Exponential smoothing per-batch loss for reporting in training progress

## [0.3.3] - 2025-01-01
- Updated docstring

## [0.3.2] - 2025-01-01
- Added progress bar for train (and test) loss evaluation
- Add options to always step the optimizer after each batch
- Added norm-first option to (repeated) skip-connection blocks
- Added Lazy concatenation option for tensors along first dimension.

## [0.3.1] - 2024-11-27
- Removed old parent directory creation logic for checkpoint file
- Added dropout option to activated block
- Toggle loss between train and eval in PyTorch training loop
- Added cross-entropy loss with label smoothing on and off for train and eval

## [0.3.0] - 2024-11-23
- Explicitly display "CLIP" in progress bar when gradient norm is clipped
- Do not create parent directory for checkpoint file until save is called
- Added parquet writer for pandas dataframes

## [0.2.9] - 2024-11-21
- Homogenized reader and writer directory handling
- Added gradient clipping to Trainer

## [0.2.8] - 2024-11-17
- Added partial to torch cat

## [0.2.7] - 2024-11-13
- Refactored helper methods in TrainData 

## [0.2.6] - 2024-11-13
- Added epoch argument to training data call method
- Changed return signature of TrainData call method
- Refactored PyTorch model trainer
- Added cosine learning-rate scheduler

## [0.2.5] - 2024-11-10
- Improved handling of missing parent directory in writers/savers
- Absorbed utility to drop None fields from into TOML writer 

## [0.2.4] - 2024-11-07
- PyTorch model trainer now reports correct loss also for "sum" reduction
- Added utility to drop None fields from dictionaries 

## [0.2.3] - 2024-11-06
- Minor change to log message
- Fixed typos in docstrings
- Added unit test to TomlWriter
- Added a PyTorch model Compile class

## [0.2.2] - 2024-11-04
- Moved optimizer stepping frequency logic into data base classes

## [0.2.1] - 2024-11-01
- Fixed checkpointing logic in PyTorch model trainer

## [0.2.0] - 2024-11-01
- Fixed stepping logic of learning rate scheduler in PyTorch model trainer

## [0.1.9] - 2024-11-01
- Changes to batch-wise warmup in PyTorch model trainer
- Changed Trainer instantiation signature
- Adjusted schedulers accordingly

## [0.1.8] - 2024-10-30
- Added missing dependency

## [0.1.7] - 2024-10-30
- Added option to accumulate gradients over multiple batches
- Learning rate scheduler now called after every epoch
- Added scheduling functions for learning rate
- Added TOML and YAML writers

## [0.1.6] - 2024-10-27
- Minor change to Pipfile
- Minor change to a unit test setup
- Added loss reporting to progress bar in training loop
- Improved use of strip in path normalization

## [0.1.5] - 2024-10-23
- No longer delete checkpoint file on instantiation in OnDisk
- Use set_to_none=True in optimizer.zero_grad

## [0.1.4] - 2024-10-16
- Added field type to resolve paths to jsonobject fields
- Added filed type to lowercase and strip strings
- Refactored PyTorch model trainer
- Refactored PyTorch data base classes
- Adapted callbacks accordingly

## [0.1.3] - 2024-10-11
- Make progress bar disappear after each epoch
- Fixed PyTorch model loader

## [0.1.2] - 2024-10-09
- Updated dependencies
- Fixed call signature of PyTorch training callbacks
- Added tqdm progress bar to PyTorch model trainer

## [0.1.1] - 2024-10-02
- Updated lock file

## [0.1.0] - 2024-10-02
- Try from pull request

## [0.0.9] - 2024-10-02
- Added missing dependency `nvitop`
- Fixed typo in README
- Added link to template repo

## [0.0.8] - 2024-10-02
- Fix documentation.

## [0.0.7] - 2024-10-02
- Check branch protection rules.

## [0.0.6] - 2024-10-02
- Added Readme.

## [0.0.5] - 2024-10-01
- Removed explicit license statement from pyproject.toml

## [0.0.4] - 2024-10-01
- Initial GitHub actions implemented.
