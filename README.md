# Karasu-Collective-Cloud-Profiling

Prototypical implementation of "Karasu" for collective and thus efficient cloud configuration profiling. 
The whole approach is implemented in [Python](https://docs.python.org/3.8/).
Please consider reaching out if you have questions or encounter problems.

## Technical Details

Strieving for understandable code that can be reused and further developed, 
we use [pydantic](https://pydantic-docs.helpmanual.io/) and
[python typing](https://docs.python.org/3/library/typing.html#) whenever feasible. 


#### Key Packages
* [BoTorch](https://botorch.org/) `0.6.0`, a framework for Bayesian Optimization in PyTorch
* [Ax](https://ax.dev/) `0.2.3`, a platform for managing and optimizing experiments
* [Hummingbird](https://microsoft.github.io/hummingbird/index.html) `0.4.3`, a library for compiling traditional ML models into tensor computations

These packages and all other required packages are specified in the `requirements.txt` and 
can thus be conveniently installed via `pip install --user -r requirements.txt` in case a normal
installation is desired. However, we recommend the containerized approach, as described next.

#### Containerization

To foster the easy deployment and execution of this prototype, we furthermore provide a `Dockerfile` for building a container image.
It can be manually built via `docker build -t karasu-container:dev .` 
Note that for your convenience, this process is already handled internally when using our bash functions.

By default, a container started with this image will simply execute a `ping` command. Further below, we describe how it can be used for actual
experiments, i.e. by overriding the default command with specific experiment / evaluation tasks. 

## Karasu in Action

We present Karasu, a collective and privacy-aware approach for efficient cloud configuration profiling. 
It trains lightweight performance models using only high-level information of shared workload profilings and combines them into an ensemble method for better exploiting inherent knowledge of the cloud configuration search space.
This way, users are able to collaboratively improve their individual prediction capabilities, while obscuring sensitive information.
Furthermore, Karasu enables the optimization of multiple objectives or constraints at the same time, like runtime, cost, and carbon footprint.

In the following, we provide instructions for reproducing our results.

#### Prerequisites

We evaluate our approach on a publicly available dataset consisting of performance data 
from diverse workloads and their various executions in a cloud environment.
Specifically, we use [this](https://github.com/oxhead/scout) dataset created in the context of proposed cloud configuration approaches.
Among other things, it encompasses data obtained from 18 workloads running 69 configurations (scaleout, VM type) in a multi-node setting (one run per configuration).
Workloads were implemented in Hadoop and different Spark versions, realized with various algorithms, and tasked with processing diverse datasets.

For our evaluation, it is required to clone this repository, and copy the folder `scout/dataset/osr_multiple_nodes` to `data/scout_multiple` in our repository.
The initial processing of this dataset will take a few minutes, depending on the concrete machine used.
It is furthermore required to mount the folder `data` as well as the to-be-created folder `artifacts` to any container you start.
This is all handled internally by the minimalistic bash functions we provide, 
so you can directly proceed with the next steps!

### Emulating a Shared Performance Data Repository

To start with, we emulate a shared performance data repository, which requires appropriate data generation using our baselines.
The hereby generated data is used in the subsequent examples for both visualizing the capabilities of individual baselines, 
and offering Karasu a data source to draw from for its ensemble approach. For single-objective optimization (SOO):
```
./docker_scripts.sh create_soo_data
```
Likewise, for multi-objective optimization (MOO):
```
./docker_scripts.sh create_moo_data
```
In our experiments, each of the executed scripts ran for approx. 10 hours. 
The generated data is saved to the `artifacts` directory and used within the next steps
where we investigate our research questions (RQs).

### RQ1: General Performance Boost

What is the general potential of exploiting existing models to boost a target one? 
We evaluate a scenario where support models are available that originate from the same workload, 
yet were initialized differently and trained with other runtime targets.
To run the experiments and generate the data for analysis, run:
```
./docker_scripts.sh run_rq1_experiment
```
In our experiments, the script ran for approx. 2 days.

### RQ2: Collaborative Applicability

How good does the introduced approach work in a collaborative scenario, with potentially diverse workloads and limited available data? 
We evaluate a scenario where all the data in the repository originates from different workloads, 
with individual characteristics, resource needs, and constraints.
To run the experiments and generate the data for analysis, run:
```
./docker_scripts.sh run_rq2_experiment
```
In our experiments, the script ran for approx. 2 days.

### RQ3: Multi-Objective Support

To evaluate Karasu in an MOO setting, we consider two objectives, 
namely cost and energy consumption, to be minimized under formulated runtime constraints.
To run the experiments and generate the data for analysis, run:
```
./docker_scripts.sh run_rq3_experiment
```
In our experiments, the script ran for more than 1 day.

### Finally: Data Analysis

With the generated data in place, one can analyze the results and produce insightful plots (as in our paper).
The plots can be created simply by running:
```
./docker_scripts.sh analysis
```
Note that running the analysis requires the completion of all aforementioned steps.

### Concluding Remarks

In our experiments, we had access to a rather modern machine equipped with a GPU (details are described in the paper).
The experiment execution still required some time (see sections above).
Now, with the execution taking place in a docker container, possibly on less sophisticated hardware,
the indicated execution times might be prolonged.

Note that it is possible to abort and resume the execution of specific experiments (Emulation, RQ1, RQ2, RQ3)
since we inspect on every container restart the already written experiment data and thus 
skip the associated configurations to prevent data duplication.

No time for data generation on your own? Consider extracting the `artifacts.tar.gz` to directly reuse our generated experiment data.

## Questions? Something does not work or remains unclear?
Please get in touch, we are happy to help!