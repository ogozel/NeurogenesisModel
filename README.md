# NeurogenesisModel

This repository contains the code associated with the manuscript at: https://www.biorxiv.org/content/10.1101/704791v4 .

## MainNetwork
This folder contains the scripts to reproduce the results of the main (continuous) rate network using the MNIST database of handwritten digits (Y LeCun, C Cortes, CJC Burges, 1999; available at: http://yann.lecun.com/exdb/mnist/).
* sim_prestore34.m: pretraining of the network - unsupervised learning of feedforward weights from randomly initialized values
* sim_neuro345.m: unresponsive units become newborn DGCs - their recurrent connections are reinitialized to random values and they have a two-step maturation process (first GABAergic input is excitatory, then it switches to inhibitory)
* sim_control_allPlastic.m: control scenario - keep all feedforward weights plastic
* sim_control_fewPlastic.m: control scenario - only unresponsive units keep plastic weights, but they do not undergo neurogenesis (no reinitialization of the recurrent connectivity weights, GABAergic input is always inhibitory)
