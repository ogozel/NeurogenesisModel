# NeurogenesisModel

This repository contains the code associated with the manuscript at: https://www.biorxiv.org/content/10.1101/704791v4 .

## MainNetwork
This folder contains the scripts to reproduce the results of the main (continuous) rate network using the MNIST database of handwritten digits (Y LeCun, C Cortes, CJC Burges, 1999; available at: http://yann.lecun.com/exdb/mnist/).
* sim_prestore34.m: pretraining of the network - unsupervised learning of feedforward weights from randomly initialized values
* sim_neuro345.m: unresponsive units become newborn DGCs - their recurrent connections are reinitialized to random values and they have a two-step maturation process (first GABAergic input is excitatory, then it switches to inhibitory)
* sim_control_allPlastic.m: control scenario - keep all feedforward weights plastic
* sim_control_fewPlastic.m: control scenario - only unresponsive units keep plastic weights, but they do not undergo neurogenesis (no reinitialization of the recurrent connectivity weights, GABAergic input is always inhibitory)

### code_analyze
This folder contains the scripts to analyze the simulated main network (at any stage).
* compute_Master_activity.m: to compute the stable DGC firing rates upon presentation of all training or testing patterns
* compute_activityProfiles.m: to determine which DGCs are activated upon presentation of example patterns, and to what extent
* analyzeFR.m: to determine the percentage of patterns that induce DGC activity in different firing rate bins
* compute_classification_error.m: to compute the classification error using the readout network. First run assign_readout_weights.m to learn the weights of the readout network.
* compute_proj_discriminatoryAxes.m: to project the stable DGC firing rates upon presentation of a pattern onto discriminatory axes
