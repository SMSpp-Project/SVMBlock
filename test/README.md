# test

The unit test of `SVMBlock`, `SVCBlock` and `SVRBlock` and of the `:Solver`
the module carries.

It generates the data set out of a seed and exercises every formulation of
the training problem, for the classification and the regression variant, for
the linear and the nonlinear kernels and for each combination of the squared
loss and of the regularised bias. What the ad hoc Solver finds is asserted by
strong duality, i.e., the value of the primal at the model recovered out of
the multipliers has to be the value the Solver reports, which checks the
Solver, the parametric map of the Block and the recovery of the model in one;
the abstract representation is held to the same number, which every
formulation shares, the Wolfe dual being written as the maximisation that
strong duality makes equal to the primal. The consensus rewriting is checked
both structurally, against what a generic Lagrangian `:Solver` asks for, and
numerically, the consensus `Constraint` having to hold at the optimum of the
monolithic problem and the objectives of the sub-`Block` having to add up to
its value. The `Solution` that saves the trained model is restored into
another Block, which then has to predict the same, directly and after a
netCDF round trip.

The comparisons that need a `:Solver` of another module are not here: they
live in the test suite of this Block.

    ./SVMBlock_test [ seed ]

The `makefile` builds the executable including this module and the core SMS++
library.


## Authors

- **Donato Meoli**  
  Dipartimento di Informatica  
  Università di Pisa


## License

This code is provided free of charge under the [GNU Lesser General Public
License version 3.0](https://opensource.org/licenses/lgpl-3.0.html) -
see the [LICENSE](LICENSE) file for details.
