# Shape Constrained Regression problem
This is a set of MATLAB functions for performing polynomial regression with shape constraints. The currently supported shape constraints are monotonicity and convexity on a bounded interval.
## Prerequisites

### MATLAB R2019a
This project was developed in MATLAB R2019a. We do not guarantee the forward
or backward compatibility to any other versions.

### YALMIP toolbox
Install YALMIP toolbox by following the instructions [here](https://yalmip.github.io/).

### MOSEK solver
Install MOSEK solver by following the instructions [here](https://www.mosek.com/downloads/).

After adding MOSEK to the MATLAB path, check the installation by running **yalmiptest**.
You should see something like that as part of the output:
~~~~
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
|                   Test|   Solution|                 Solver message|
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
|   Core functionalities|        N/A|   Successfully solved (YALMIP)|
|                     LP|    Correct|    Successfully solved (MOSEK)|
|                     LP|    Correct|    Successfully solved (MOSEK)|
|                     QP|    Correct|    Successfully solved (MOSEK)|
|                     QP|    Correct|    Successfully solved (MOSEK)|
~~~~
As a note, make sure to download the MOSEK license file and place it in the `\mosek\` folder.

*As a note! You might experience issues in the installation process, I found the YALMIP [forum](https://groups.google.com/forum/#!forum/yalmip) to be very prompt and helpful at answering questions.*


## Monotone Regression
The file [monotone_regression.m](monotone_regression.m) implements the main function for polynomial regression with monotonicity constraints.

## Bounded Derivative Regression
The file [bounded_derivative_regression.m](bounded_derivative_regression.m) implements the main function for polynomial regression with bounded-derivative constraints. Bounded Derivative Regression generalizes Monotone Regression, which has only one-sided bounds. It also generalizes Lipchitz Regression where the bounds on the derivative are symmetric in the positive and negative directions.

## Convex Regression
The file [convex_regression.m](convex_regression.m) implements the main function for polynomial regression with joint convexity constraints.

## Monotone-Convex Regression
The file [monotone_convex_regression.m](monotone_convex_regression.m) implements the main function for polynomial regression with joint convexity constraints as well as monotonicity constraints.

## Experiments
