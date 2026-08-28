# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LIBSVMSolver`, which hands the training problem over to LIBSVM: an
  independent implementation of the very algorithm `SMOSolver` implements,
  hence a reference to check it against, and a fast one for a large data set.
  It reads the physical representation, recovers the multipliers, the bias and
  the value out of the kernel expansion LIBSVM returns, and refuses the
  training problems LIBSVM cannot express rather than approximating them. It
  is built only when LIBSVM is found

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

- `SMOSolver::get_Solution()`, which builds the SVMBlockSolution out of the
  multipliers the Solver holds, without writing anything into the SVMBlock and
  therefore without requiring any Variable to exist; any other Solution saves
  the abstract representation, and is left to the base class. Requires the
  `Solver::get_Solution()` of the core

- `SVMBlockSolution`, the Solution saving the trained model rather than the
  abstract representation, which is what `get_Solution()` returns when the
  Configuration asks for it and what makes the model writable to a file

### Changed

- the consensus rewriting is a *structure* of the `SVMBlock`, chosen by the
  new `set_structure()` out of the number of chunks, rather than a separate
  Block that a free function assembles: `make_consensus_Block()` is therefore
  gone, and with it the `AbstractBlock` that used to be the father. Since the
  `SVMBlock` still holds the whole data set, a Solver reading the physical
  representation keeps solving the very same training problem whatever the
  structure, so that `SMOSolver` can now be attached to the rewriting as well

- the Gram matrix of a data set large enough to be worth a thread is computed
  in parallel

### Fixed

- `SVMBlockSolution::read()` only looked at the physical representation, hence
  it saved nothing at all when the SVMBlock had been solved by a Solver
  working on the abstract one, which is where the model is then left

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
