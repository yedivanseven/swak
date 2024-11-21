# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
