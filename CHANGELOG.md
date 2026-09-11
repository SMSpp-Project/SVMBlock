# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LIBLINEARSolver`, which hands the training problem over to LIBLINEAR: the
  linear kernel only and the regularised bias only, but both losses and any
  weight of the regularisation term, which is the side of the problem
  `LIBSVMSolver` does not cover. A run that stops on the cap LIBLINEAR has on
  its own iterations is reported as `kLowPrecision`

- `get_K_row()`, which serves one row of the Gram matrix, and
  `set_K_memory()`, which says how much memory the matrix may take: under
  that budget the whole matrix is built as before, over it the rows are
  computed on demand and kept in a cache of the least recently used ones, so
  that a data set whose Gram matrix does not fit in memory can be trained
  anyway. `set_K_active()` tells the cache which entries are going to be read,
  and each row records which of its own it has already computed: an algorithm
  that shrinks its active set therefore pays for the entries it asks for, and
  completes a row rather than recomputing it when the set grows back

- `add_samples()` and `remove_samples()`, which change the data set instead of
  replacing it, with the `eAddSamples` and `eRmvSamples` Modification: the
  samples that stay are not touched, their multipliers keep their value and
  the Gram matrix is extended or compacted rather than recomputed. Whatever is
  indexed over the dual index space is dynamic accordingly, and the abstract
  representation is extended and shrunk rather than rebuilt. `SMOSolver`
  re-optimizes across them, which is what an incremental training, and a
  k-fold cross-validation done by taking a fold out and putting it back, need

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

- the Gram matrix is filled by as many threads as the data set warrants, one
  every few hundred rows instead of one per core, and they are started once
  and reused: on a machine with hundreds of cores starting them costs more
  than filling the matrix does, and a training that builds many models paid
  it once per model

- the two dual indices of a sample of a `SVRBlock` are *adjacent*, rather than
  the two sides of the tube being one block each: a sample then adds its
  multipliers at the end of the dual index space, which is where a dynamic
  `Variable` is added. Note that this changes the order in which the
  multipliers are stored, hence that of a `SVMBlockSolution` written to a file

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

- the cache of the rows of the Gram matrix did not survive a change of the
  data set: it was laid out once on the number of samples of the moment, so
  adding samples wrote past the end of it, and its mask marked as present the
  bits past the end of a row, which stand for no entry until the row grows
  into them and then serve an entry that was never computed. Learning one
  sample at a time under a memory budget therefore either crashed or took a
  different number of iterations; it now takes the same number as with the
  whole matrix, which is the only right answer

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
