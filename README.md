# Dantzig Project
[Add project description]
## Prerequisites 

### MATLAB R2017b
This project was developed in MATLAB R2017b. We do not guarantee the forward
or backward compatibility to any other versions.

### YALMIP toolbox
Install YALMIP toolbox by following the instructions [here](https://yalmip.github.io/).

### MOSEK solver
Install MOSEK solver by following the instructions [here](https://www.mosek.com/downloads/).

After adding MOSEK to the MATLAB path, check the installation by running **yalmiptest**.
You should see something like that as part of the oiutput:
~~~~
|         MOSEK|             SOCP|       found|
|         MOSEK|            LP/QP|       found|
|         MOSEK|              SDP|       found|
|         MOSEK|        GEOMETRIC|       found|
~~~~