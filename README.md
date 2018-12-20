Machine Learning Molecular Dynamics (mlmd)
==========================================

The machine learning molecular dynamics (mlmd) package is a command line open source python program. It calculates the feature representation of a structure (molecule, cluster, crystal), using the Filtered Beher Parrinello (FBP), in addition,  mlmd is also able to perform molecular dynamics simulations with machine learning potentials (also known as machine learning force fields or machine learning empirical potentials). 

The mlmd program generates machine learning potentials (mlp) processing the energies and forces of DFT calculations carried out over systems of particles, then uses the trained potential to do NVT molecular dynamics on a structure compatible with the mlp. So far mlmd can use calculations from [VASP](https://www.vasp.at/index.php/about-vasp/59-about-vasp), [Abinit](https://www.abinit.org/), and [Fireball](https://sites.google.com/site/jameslewisgroup/) as inputs.

mlmd requirements
-----------------
For its correct functioning mlmd needs the following codes:

1. [Python](https://www.python.org/download/releases/2.7/ "Python") = 2.7

2. [Numpy](http://www.numpy.org/ "Numpy") >= 1.1 1

3. [sklearn](https://scikit-learn.org/stable/)

4. [Tensorflow](https://www.tensorflow.org/)

* [Pychemia](https://github.com/MaterialsDiscovery/PyChemia) In case you need to read data from VASP

mlmd installation
-----------------


# Using mlmd

Training (creating) a machine learning potential
------------------------------------------------

Contributors
------------
* Prof. James P. Lewis [West Virginia University] 
* Prof. Aldo H. Romero [West Virginia University] 

* Arturo Hernandez [West Virginia University] (Developer)

* Uthpala Herath   [West Virginia University] (Simulation and testing) 

* Pedram Tavazohi  [West Virginia University] (Simulation and testing)
