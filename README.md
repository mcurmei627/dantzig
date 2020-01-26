# Shape Constrained Regression problem
[Add project description]
## Prerequisites 

### MATLAB R2017b
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

## Monotone Regression
[Add reference to monotone_regression.m file]

## Convex Regression

## Experiments
### Artificial Data
#### Monotone regression
[Explain setup and the true underlying function]
#### Convex Regression

