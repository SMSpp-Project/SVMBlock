# SVMBlock

Definition and implementation of the `SVMBlock` class, which implements the
`Block` interface within the SMS++ framework for the training problem of a
*Support Vector Machine* (SVM), together with the ad hoc `SMOSolver`.

Two concrete classes derive from the abstract `SVMBlock`: `SVCBlock`, for the
*Support Vector Classifier*, i.e., the maximum-margin hyperplane separating the
samples of two classes, and `SVRBlock`, for the *Support Vector Regression*,
i.e., the model whose errors are not penalised inside a tube of given
half-width around the targets. Both support the linear and the squared
penalisation of the training errors, i.e., the hinge and the squared hinge loss
for classification and the epsilon-insensitive and the squared
epsilon-insensitive loss for regression, and both can either keep the bias out
of the regularisation term or fold it into the weight vector.

The `Block` holds the data set, the hyper-parameters and the kernel, and its
abstract representation can encode either of two problems, selected through a
`Configuration`:

- the **training problem** itself, a quadratic program in the weights, the
  bias and the training errors, whose Hessian is *diagonal* and whose
  constraints are linear, so that it is directly handled by any
  general-purpose quadratic `Solver`; it is only available for the linear
  kernel, which is the only one whose feature map is the identity;

- its **Wolfe dual**, a quadratic program in the multipliers over a box with
  (at most) one linear equality constraint, whose Hessian is the Gram matrix of
  the kernel reweighted by the labels. It never involves the features
  explicitly, which is what makes the nonlinear kernels possible and what makes
  it the *only* thing there is for them. It is written as the maximisation it
  customarily is, so that strong duality, which holds since the training
  problem is convex, makes the value of the `Objective` the very same number
  whichever of the two is encoded.

Writing the training problem as one problem per chunk of samples, each with its
own copy of the model and an even share of the regularisation term, the copies
tied together by linear *consensus* constraints, is a way of *solving* it
rather than a property of it: a data set has no structure of its own, and the
number of chunks is a choice of whoever solves. It is therefore not something
the `Block` encodes but something `make_consensus_Block()` assembles out of
one, giving exactly the structure `LagrangianDualSolver` expects, so that
relaxing the consensus constraints turns the training problem into one
independent, and much smaller, SVM per chunk; equivalently, it is its
Dantzig-Wolfe decomposition over the chunks. Like the primal, it needs the
linear kernel.

The linear, polynomial, gaussian, laplacian and sigmoid kernels are provided.
Whichever formulation and `Solver` is used, the trained model is available in
the kernel expansion form, and for the linear kernel the weight vector is
available as well.

`SMOSolver` implements the `Solver` interface for a `SVMBlock` with the
*Sequential Minimal Optimization* algorithm on the dual, i.e., the
decomposition method that at each iteration optimizes over the smallest
possible working set and therefore never needs the (dense) Hessian of the dual
as a whole, which is what makes it the standard choice for training a SVM. It
reads the data out of the physical representation of the `SVMBlock`, so it
does not require the abstract one to be generated at all. The two-multiplier
step is the one of

J. C. Platt "Sequential Minimal Optimization: A Fast Algorithm for Training
Support Vector Machines" *Microsoft Research technical report* MSR-TR-98-14,
1998

while the working set is selected, and the algorithm is stopped, with the two
thresholds of

S. S. Keerthi, S. K. Shevade, C. Bhattacharyya, K. R. K. Murthy "Improvements
to Platt's SMO Algorithm for SVM Classifier Design" *Neural Computation* 13(3),
637-649, 2001

which for the regression case reduce to the thresholds of

S. K. Shevade, S. S. Keerthi, C. Bhattacharyya, K. R. K. Murthy "Improvements
to the SMO Algorithm for SVM Regression" *IEEE Transactions on Neural Networks*
11(5), 1188-1193, 2000

since a regression sample contributes two multipliers with opposite signs,
whence the two sides of its insensitivity tube.

Solving the consensus rewriting needs modules this one does not depend on: the
`SVMBlock` suite of the [tests](https://gitlab.com/smspp/tests) repo does it
with `LagrangianDualSolver`, `BundleSolver` and a `:MILPSolver`, and the
`svm_solver` of the [tools](https://gitlab.com/smspp/tools) repo trains a model
and performs the model selection around it.


## Getting started

These instructions will let you build the `SVMBlock` module on
your system.

### Requirements

- The [SMS++ core library](https://gitlab.com/smspp/smspp) and its
  requirements.

### Build and install with CMake

Configure and build the library with:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```

The library has the same configuration options of
[SMS++](https://gitlab.com/smspp/smspp-project/-/wikis/Customize-the-configuration).

Optionally, install the library in the system with:

```sh
cmake --install .
```

### Usage with CMake

After the library is built, you can use it in your CMake project with:

```cmake
find_package(SVMBlock)
target_link_libraries(<my_target> SMS++::SVMBlock)
```

### Build and install with makefiles

Carefully hand-crafted makefiles have also been developed for those unwilling
to use CMake. Makefiles build the executable in-source (in the same directory
tree where the code is) as opposed to out-of-source (in the copy of the
directory tree constructed in the build/ folder) and therefore it is more
convenient when having to recompile often, such as when developing/debugging
a new module, as opposed to the compile-and-forget usage envisioned by CMake.

Each executable using `SVMBlock` has to include a "main makefile" of
the module, which typically is either [makefile-c](makefile-c) including all
necessary libraries comprised the "core SMS++" one, or
[makefile-s](makefile-s) including all necessary libraries but not the "core
SMS++" one (for the common case in which this is used together with other
modules that already include them). These in turn recursively include all the
required other makefiles, hence one should only need to edit the "main
makefile" for compilation type (C++ compiler and its options) and it all
should be good to go. In case some of the external libraries are not at their
default location, it should only be necessary to create the
`../extlib/makefile-paths` out of the `extlib/makefile-default-paths-*` for
your OS `*` and edit the relevant bits (commenting out all the rest).

Check the [SMS++ installation wiki](https://gitlab.com/smspp/smspp-project/-/wikis/Customize-the-configuration#location-of-required-libraries)
for further details.


## Getting help

If you need support, you want to submit bugs or propose a new feature, you
can [open a new issue](https://gitlab.com/smspp/svmblock/-/issues/new).


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct, and the process for submitting merge requests to us.


## Authors

### Current Lead Authors

- **Donato Meoli**  
  Dipartimento di Informatica  
  Università di Pisa

### Contributors


## License

This code is provided free of charge under the [GNU Lesser General Public
License version 3.0](https://opensource.org/licenses/lgpl-3.0.html) -
see the [LICENSE](LICENSE) file for details.


## Disclaimer

The code is currently provided free of charge under an open-source license.
As such, it is provided "*as is*", without any explicit or implicit warranty
that it will properly behave or it will suit your needs. The Authors of
the code cannot be considered liable, either directly or indirectly, for
any damage or loss that anybody could suffer for having used it. More
details about the non-warranty attached to this code are available in the
license description file.
