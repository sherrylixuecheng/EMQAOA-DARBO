# EMQAOA-DARBO

<p align="center">
  <a href="https://github.com/sherrylixuecheng/EMQAOA-DARBO">
    <img width=80% src="https://github.com/sherrylixuecheng/EMQAOA-DARBO/blob/main/schematic.png">
  </a>
</p>

## Overview
This repository includes the codes and results for the manuscript:
***Error-mitigated Quantum Approximate Optimization via Learning-based Adaptive Optimization***

## Installation and usage
This repository requires to install two open-sourced packages: 

* [ODBO](https://github.com/tencent-quantum-lab/ODBO) packge: The installation direction is provided in the corresponding main page.

* [TencirCircuit](https://github.com/tencent-quantum-lab/tensorcircuit) or TC: ```pip install tensorcircuit``` 

** To enable the usage of TC-Hardware inferface to run hardware experiments, you need to use [TencirCircuit-dev](https://github.com/refraction-ray/tensorcircuit-dev) 

## Content list

### Files

* [DARBO_optimization_ideal_example.ipynb](DARBO_optimization_ideal_example.ipynb): This is a simple example to illustrate the methods \& to run a test MAX-CUT on a random graph with a circuit depth of 4.

* [EMQAOA_DARBO_run.ipynb](EMQAOA_DARBO_run.ipynb.ipynb): This is the notebook to illustrate the EMQAOA-DARBO on the real hardware. This collects the hardwared data shown in the manuscript. Note: For non-Tencent-Quantum-Lab user, this set of codes cannot be run directly due to the unavailable access to the Tencent hardware. If you would like to have a try, please contact Tencent Quantum Lab to check the possible options for usage. 

* [si_more_stats.xlsx](si_more_stats.xlsx): This is a supplemental excel to summarize the optimized losses and $r$ values for different optimizers and different cases.


### Folders

* [codes](codes): contains all the python codes that run the experiments collected in this work. (Please aware that all BO methods are formulated as a maximization problem (```max -loss```), and we save the ```-loss``` at each iteration. For other optimizers, we save ```loss``` at each iteration.)

* [graph](graph): contains the graphs used in this work.

* [initialization](initialization): contains the presaved (\& different) initialized parameters to make sure all different optimizers running from the same initial guesses.

* [results](results): each subfolder contains the collected results for the corresponding 

* [plotting](plotting): contains a jupyter notebook to generate all the plots used in the paper. [for_plotting](plotting/for_plotting) folder contains the .txt summary for the results extracted from the raw results. 


## Please cite us as

```
@article{cheng2023darbo,
  title={Error-mitigated Quantum Approximate Optimization via Learning-based Adaptive Optimization},
  author={Cheng, Lixue and Chen, Yu-Qin and Zhang, Shi-Xin and Zhang, Shengyu},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2023}
}
```
