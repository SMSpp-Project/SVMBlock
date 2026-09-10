# SVMBlock

Definition and implementation of the `SVMBlock` class, which implements the
`Block` interface within the SMS++ framework for the training problem of a
*Support Vector Machine* (SVM), together with the ad hoc `SMOSolver` and with
`LIBSVMSolver` and `LIBLINEARSolver`, which hand the training problem over to
LIBSVM and to LIBLINEAR.

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

The training problem can also be written as one problem per chunk of samples,
each with its own copy of the model and an even share of the regularisation
term, the copies tied together by linear *consensus* constraints. Which of the
two the `Block` is, i.e., whether it has sub-`Block` at all, is a *structure*
it is given, and it is chosen by `set_structure()` out of a
`SimpleConfiguration< int >`, the number of chunks: with more than one the
`SVMBlock` has one sub-`SVMBlock` per chunk and the consensus constraints are
the only ones it has of its own, it having no `Variable` at all. That is
exactly the structure `LagrangianDualSolver` expects, so that relaxing the
consensus constraints turns the training problem into one independent, and
much smaller, SVM per chunk; equivalently, it is its Dantzig-Wolfe
decomposition over the chunks. Like the primal, it needs the linear kernel.
Note that the `SVMBlock` still holds the whole data set, so a `Solver` reading
the physical representation, such as `SMOSolver`, keeps solving the very same
training problem whatever the structure.

Samples can also be *added* to and *removed* from the data set at any time,
which is what an incremental training and a k-fold cross-validation done by
taking a fold out and putting it back need [see `add_samples()` and
`remove_samples()`]. The samples that are already there are not touched: their
multipliers keep their value, and what the Gram matrix already holds is kept,
only the entries of the new samples being computed. Whatever is indexed over
the dual index space is therefore *dynamic*, i.e., the multipliers and their
bounds in the dual, the slacks with their bounds and the margin constraints in
the primal; the weights and the bias are indexed over the features, which do
not change, and are static. The abstract representation is *extended* and
*shrunk* rather than rebuilt, so that a `Solver` reading it also has only to
deal with what has actually changed: the multipliers a sample adds come at the
end of the dual index space, which is where a dynamic `Variable` is added, and
that is why the two multipliers of a regression sample are adjacent rather
than the two sides of the tube being one block each.

The hyper-parameters and the targets can be changed at any time, also while a
`Solver` is attached and the abstract representation is constructed: the
`Block` updates the latter and issues both the *physical* `Modification`
saying what exactly has changed and the *abstract* ones describing how the
abstract representation has changed as a consequence, so that a `Solver`
reading either representation can react to it. Whatever changes the Hessian of
the dual as a whole, i.e., the kernel, the data set or the regularisation of
the bias, rather rebuilds the abstract representation and issues a
`NBModification`, since there would be no point in describing such a change
term by term.

The linear, polynomial, gaussian, laplacian and sigmoid kernels are provided.
Whichever formulation and `Solver` is used, the trained model is available in
the kernel expansion form, and for the linear kernel the weight vector is
available as well. `SVMBlockSolution` saves that model, which is what outlives
the training problem and is therefore what one writes to a file; it is not
what `get_Solution()` returns by default, since a `Solver` working on the
abstract representation, and the machinery combining solutions of sub-`Block`,
need to see the latter instead.

`SMOSolver` implements the `Solver` interface for a `SVMBlock` with the
*Sequential Minimal Optimization* algorithm on the dual, i.e., the
decomposition method that at each iteration optimizes over the smallest
possible working set and therefore never needs the (dense) Hessian of the dual
as a whole, which is what makes it the standard choice for training a SVM. It
reads the data out of the physical representation of the `SVMBlock`, so it
does not require the abstract one to be generated at all, and it provides the
trained model directly as a `SVMBlockSolution`, without writing it into the
`Block` first and therefore without needing any `Variable` to write it into.
It also keeps the multipliers and the gradient of the dual at them across the
calls to `compute()`, and reads the `Modification` the `SVMBlock` issues to
find out what to do with them: whatever leaves the Hessian of the dual alone,
such as the trade-off parameter or the half-width of the insensitivity tube,
only
requires the multipliers to be scaled back into their bounds and the gradient
to be updated in linear time, so that the re-optimization starts from the
previous solution and a model selection costs much less than the sum of the
individual trainings. Samples being added or removed is followed as well: the
multipliers of those that are still there are kept, matched to the dual index
space of the new data set by the sample and the sign they refer to, those of
the new samples start at zero, the equality constraint is made to hold again
by giving back to the multipliers that can absorb it whatever a removal has
left in excess, and the gradient is recomputed with one pass over the Gram
matrix, which has been extended or compacted rather than recomputed. The two-multiplier step is the one of

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

`LIBSVMSolver` hands the training problem over to
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), the reference
implementation of the very algorithm `SMOSolver` implements: it is therefore
an independent implementation to check the latter against on any data set, and
a mature and fast one when the data set is large. Like `SMOSolver` it reads
the physical representation, and it recovers the multipliers, the bias and the
value of the training problem out of what LIBSVM returns, which is the kernel
expansion of the model. LIBSVM solves a smaller family of training problems
than a `SVMBlock` can encode: the loss has to be the linear one, the bias has
to be out of the regularisation term, the weight of that term has to be 1 and
the kernel cannot be the Laplacian one. Anything else is refused rather than
approximated, since the value and the model this `Solver` reports are meant to
be compared with those of the ones that solve the very problem. It is built
only when LIBSVM is found, everything else in the module requiring nothing
beyond the core.

`LIBLINEARSolver` hands the training problem over to
[LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/), which trains a
linear model by working in the weights rather than in the multipliers, and it
covers the side of the problem LIBSVM does not: both losses and any weight of
the regularisation term, but the linear kernel only and the regularised bias
only, the bias being one more feature of value one appended to each sample.
What it returns is the model rather than the kernel expansion, so the value of
the training problem is computed here out of the model; and since the number
of iterations of LIBLINEAR is capped in its own sources rather than being a
parameter, a run that stops on the cap is reported as `kLowPrecision`, its
value being an upper bound on the optimal one and nothing more. It is built
only when LIBLINEAR is found.

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

- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), optional and only
  needed by `LIBSVMSolver`, which is left out of the library when it is not
  found. Any version from 3.0 on does: `sudo apt install libsvm-dev` on
  Debian/Ubuntu, `brew install libsvm` on macOS, `vcpkg install libsvm` on
  Windows.

- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/), optional and
  only needed by `LIBLINEARSolver`, which is left out of the library when it
  is not found. Any version from 2.0 on does: `sudo apt install
  liblinear-dev` on Debian/Ubuntu, `brew install liblinear` on macOS,
  `vcpkg install liblinear` on Windows.

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
