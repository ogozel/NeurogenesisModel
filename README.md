# NeurogenesisModel

This repository contains the code associated with the article "A functional model of adult dentate gyrus neurogenesis", eLife 2021;10:e66463. DOI: https://doi.org/10.7554/eLife.66463.

## MainNetwork
This folder contains the scripts to reproduce the results of the main (continuous) rate network using the MNIST database of handwritten digits (Y LeCun, C Cortes, CJC Burges, 1999; available at: http://yann.lecun.com/exdb/mnist/). First, create a folder ‘dataset_mnist’ containing ‘mnist144.mat’, a file which contains 12x12 downsampled images from the original 28x28 images by applying a 2x2 mean kernel over all pixels except the two pixels on the borders.
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

## SimpleNetwork
This folder contains the scripts to reproduce the results of the simplified (binary) rate network using a hand-made artificial dataset, as well as the scripts to reproduce the artificial datasets themselves.
### dataset_handmade
This folder contains the scripts to create the artificial dataset. Run ‘create_dataset.m’ with the wanted parameters, and save the resulting file in the folder.
* sim_neurogenesis.m: to learn the feedforward weights of the simple network using the artificial dataset. First, pretraining with only two DGCs, then a newborn DGC is introduced and undergoes a two-step maturation. Receptive fields are plotted, as well as the evolution of the norm of the feedforward weight vector onto the newborn DGC and its angle with the center of mass of the novel cluster.
