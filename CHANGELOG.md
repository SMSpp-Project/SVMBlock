# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SVMBlockSolution`, the Solution saving the trained model rather than the
  abstract representation, which is what `get_Solution()` returns when the
  Configuration asks for it and what makes the model writable to a file

## [0.1.0] - 2026-08-04

### Added

- `SVMBlock`, the abstract base class holding the data set, the
  hyper-parameters, the kernel and both the training problem and its Wolfe
  dual, the latter written as the maximisation it customarily is, so that
  the value of the Objective is the same number whichever is encoded

- `SVCBlock` and `SVRBlock`, the classification and the regression variants

- `SMOSolver`, the ad hoc Sequential Minimal Optimization solver for the dual

- `make_consensus_Block()`, which assembles the training problem written as
  one problem per chunk of samples tied by consensus constraints, i.e., the
  structure a generic Lagrangian, or Dantzig-Wolfe, Solver attacks

- netCDF and text serialization, and the tester
