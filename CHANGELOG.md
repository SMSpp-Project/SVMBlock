# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SVMBlockMod`, `SVMBlockRngdMod` and `SVMBlockSbstMod`, the Modification
  describing a change of the training problem, and the abstract Modification
  issued alongside them; the hyper-parameters can therefore now be changed
  while the abstract representation is constructed and a Solver is attached,
  and so can the targets, through the new `chg_target()` and `chg_targets()`

- `SMOSolver` re-optimizes: it keeps the multipliers and the gradient of the
  dual at them across the calls to `compute()` and, whenever the change the
  Modification describe leaves the Hessian alone, it scales the former back
  into their bounds and updates the latter in linear time rather than
  restarting from the origin

- `SVMBlockSolution`, the Solution saving the trained model rather than the
  abstract representation, which is what `get_Solution()` returns when the
  Configuration asks for it and what makes the model writable to a file

### Changed

- the Gram matrix of a data set large enough to be worth a thread is computed
  in parallel

### Fixed

- loading a new data set left the SVMBlock with no abstract representation at
  all, so that a Solver reading it found an empty problem after the
  NBModification; it is now rebuilt out of the new data set

### Fixed

- the conventional values of gamma, which are derived from the data set, were
  derived again at each of the O( n^2 ) entries of the Gram matrix

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
