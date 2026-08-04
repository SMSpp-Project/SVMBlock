# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-04

### Added

- `SVMBlock`, the abstract base class holding the data set, the
  hyper-parameters, the kernel and both formulations of the training problem

- `SVCBlock` and `SVRBlock`, the classification and the regression variants

- `SMOSolver`, the ad hoc Sequential Minimal Optimization solver for the dual

- the decomposed formulation, i.e., the consensus reformulation over chunks of
  samples that a generic Lagrangian, or Dantzig-Wolfe, Solver can attack

- netCDF and text serialization, and the tester
