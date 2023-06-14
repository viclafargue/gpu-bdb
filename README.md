Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
This file is part of a software package distributed by the TPC
The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
license agreements.
This file is subject to the terms and conditions outlined in the End-User
License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.

Copyright 2021 Intel Corporation.
This software and the related documents are Intel copyrighted materials, and your use of them
is governed by the express license under which they were provided to you ("License"). Unless the
License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or
transmit this software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or implied warranties,
other than those that are expressly stated in the License.

# The RAPIDS GPU Express Benchmark AI

This benchmark repository is derived and from the TPC Express Benchmark AI (TPCx-AI) benchmark. It was modified to be used within the context of RAPIDS software, results cannot be directly compared to those obtained with the unmodified TPCx-AI benchmark reference implementation.
TPC Express Benchmark AI (TPCx-AI) is an AI benchmark that models several aspects of an AI or machine learning data science pipeline using a diverse dataset that includes images, text and audio. The benchmark defines and provides a means to evaluate the System Under Test (SUT) performance as a general-purpose data science system.

Two reference implementations are provided with this kit. The general process to enable the execution of the benchmark consists of the following steps:
1. Setting up the execution environment: Involves installing and configuring the hardware (e.g. CPU, memory, network interfaces, etc.) and software (e.g. Operating system, drivers, compilers, system administration tools, etc.) to an optimal state for the execution of the benchmark.
2. Setting up the benchmark: Includes installing dependencies needed by the kit to successfully run in your execution environment. This may include creating virtual environments, installing direct software dependencies for the kit, copying files to multiple nodes in the case of a cluster, etc. Additional details are provided in the *Setting up the benchmark* section below.
3. Creating a benchmark configuration file: Involves editing one of the configuration files included in the kit to provide configuration parameters that optimize the usage of your system resources. The *driver/config* directory contains two sample configuration files: default.yaml and default-spark.yaml that can be used as configuration start point for the single-node and the multi-node implementations, respectively.
Changes to those configuration files are in general allowed. However, parameters that can be used to change the runtime characteristics of the use cases  (e.g. num_clusters, epochs), typically specified in the *training_template*, *serving_template*, and *serving_throughput_template* keys, have restrictions and cannot be changed unless specified in Appendix C of the specification. Additional details are provided in the *Running the benchmark* and the *Additional details for the multi-node reference implementation* sections below.
4. Running the benchmark: Execute all benchmark phases. More details on how to run the benchmark are provided in the *Benchmark workflow* and the *Running the benchmark* sections below.

# Important
**This benchmark is intended to be executed in test systems only, NOT in production systems.**

This document describes the TPCx-AI benchmark kit installation and execution on test machines used during the development and validation of the benchmark. The initial state of your system (e.g. Operating system distribution and version, software packages selected during the installation of the OS, etc.) might require you to execute additional steps that are beyond the scope of this document.

# Setting up the benchmark
Please follow the steps below to setup and run either the single-node implementation or the multi-node implementation.  We'll refer to the path where the benchmark root directory is located as <TPCx-AI_HOME>. For instance, if <TPCx-AI_HOME> is /opt/tpcx-ai, then the TPCx-AI_Benchmarkrun.sh file is located in /opt/tpcx-ai/run_benchmark.sh.

Notes:
  * The setup process will install software packages listed in the .yml files included in the subidrectories of <TPCx-AI_HOME>/tools/. The .yml files included with the kit have versions that are known to work based on our tests and should be used for reference. Other versions (older or newer) of the same packages may or may NOT work. All versions of the packages need to comply with the software requirements (e.g. versioning, support, pricing, etc.) described in the specification document.
  In addition, the jar files included with the kit under the <TPCx-AI_HOME>/lib directory, excluding those under the <TPCx-AI_HOME>/lib/pdgf/ subdirectory, can be replaced for newer versions of the same package as long as they comply with the software requirements (e.g. versioning, support, pricing, etc.) described in the specification document.
  * The troubleshooting section below describes problems, workarounds, and limitations to common issues that can be found during the setup of the benchmark or its execution.


## Single node ##

**Prerequisites for single node reference implementation (python)**
* Runtime
  * Python 3.6+
* Tools
  * Anaconda3/Conda4+ in all nodes
* Libraries
  * libglvnd-glx
  * libsndfile
  * libsndfile-devel
* The binaries "java", "sbt" and "conda" must be included (and have "priority") in the PATH environment variable.
* Disk space: Make sure you have enough disk space to store the data that will be generated for your test. By default, the dataset will be generated in the "output/raw_data" subdirectory. The value of "TPCxAI_SCALE_FACTOR" in the setenv.sh file will determine the approximate size(GB) of the dataset that will be generated and used during the benchmark execution.

cd to <TPCx-AI_HOME> and run the  setup-python.sh  script. The script will set up the modules included in the kit and install any required prerequisites for running the single node reference implementation.

## Multi-node ##

**Prerequisites for multinode implementation (Spark)**
* CDP7.1+ (Services required: HDFS, YARN, Spark 2.4+)
* Runtime
  * Java 8 (JRE)
  * Python 3.6+
* Tools
  * Anaconda3/Conda4+ in all nodes
  * cmake
  * GCC 9.0 in all nodes
  * OpenMPI 4.0+ in all nodes
  * pssh
* Libraries
  * libglvnd-glx
  * libsndfile
  * libsndfile-devel
  * libXxf86vm
  * libXxf86vm-devel
  * mesa-libGL-devel
* All prerequisite binaries, such as "java", "sbt" and "conda" must be included (and have "priority") in the PATH environment variable.
* Disk space: Make sure you have enough disk space to store the data that will be generated for your test. By default the dataset will be generated in the "output/raw_data" subdirectory and uploaded to the "output" subdirectory in HDFS. Please backup your data if you are already using these directories for something else. The value of "TPCxAI_SCALE_FACTOR" in the setenv.sh file will determine the size(GB) of the dataset that will be generated and used during the benchmark execution.
* Create a "nodes" file in TPCx-AI_HOME. The file contains  the list of worker nodes (one node per line)  in the Spark cluster

cd to <TPCx-AI_HOME> and run the setup-spark.sh script. The script will build the benchmark binaries for running the multi-node reference implementation.

In the case of the multi-node (Spark) implementation, in addition to running the setup-spark.sh script, the setup for the execution of the deep learning use cases is more involved than it is for the traditional machine learning, typically (not necessarily) requiring the creation of a python virtual environment in all nodes with several python packages and their dependencies.
We have included in the kit a series of files and scripts that could help with the setup of such virtual environment. The scripts worked in our test systems and are not guaranteed to work in all cases. However, they should be easy to adapt to other execution environments. The following steps can be used to create such virtual environments:

### Manual setup ###

**- You are required to execute the same steps in all the nodes in your cluster. It's recommended to use a tool like pssh to run the instructions in all the nodes at the same time**

**- We continue looking for simpler ways to help users setup their execution environments and will update these steps, files and utilities as we find better/easier ways to enable the benchmark execution**

** Setting up the Python virtual environment for the DL use cases **
The file tools/spark/build_dl.yml can be used to create a virtual environment with Conda for the execution of the deep learning use cases.
The following sequence of commands have worked in our test environment:
* conda install gcc_linux-64 gxx_linux-64
* conda create -y -p /usr/envs/adabench_dl python=3.7; conda env update -p /usr/envs/adabench_dl --file tools/spark/build_dl.yml

We include a *tools/spark/horovod_test_yarn.py* script that can help confirm the correct setup of horovod in your cluster. The script is located in the *tools/spark* subdirectory and can be run as follows: *spark-submit tools/spark/horovod_test_yarn.py*. You may specify values for the number of Spark executors and the memory per executor so that at least one executor is spawned in each node.


# Benchmark workflow

The benchmark comprises 10 use cases that implement different AI or machine learning data science pipeline. The details of each use case can be found in the benchmark specification document.
The TPCx-AI tests are initiated by the TPCx-AI_Validation.sh and the TPCx-AI_Benchmarkrun.sh master scripts which can be executed from <TPCx-AI_HOME> and run all phases of the benchmark. For a TPCx-AI benchmark run to be valid, the user needs to execute first the TPCx-AI_Validation.sh script and then the TPCx-AI_Benchmarkrun.sh run script. The TPCx-AI_Validation.sh script will run all phases of the benchmark using scale factor 1; its purpose is to confirm that the minimum requirements for the benchmark execution are in place and check whether the SUT is in a reasonably good initial state for the execution of a full benchmark run using a larger scale factor.

The tests or phases performed by the benchmark are the following:
- Data generation: The process of using PDGF to create the data in a format suitable for presentation to the load facility.
- Load Test: Copying the input dataset files to the final location from where they will be eventually accessed to execute each one of the use cases.
- Power Training Test: Determines the maximum speed the SUT can process the Training of all 10 use cases by running their training phase sequentially (i.e. uc1 training, uc2 training, ..., uc10 training).
- Power Serving Test I and II: Determines the maximum speed the SUT can process the Serving stages of all 10 use cases by running their serving phase sequentially (i.e. uc1 serving, uc2 serving, ..., uc10 serving).
- Scoring: Determines the accuracy metric or error incurred by the use case. It runs the serving phase of each use case for a small data set and compares the predictions against pre-generated ground truth labels or values.
- Throughput Serving Test: Runs the serving stage of all 10 use cases using concurrent streams.
- Verification: Determines whether the accuracy/error metric of each use case meets or exceeds its predefined threshold.

# Data redundancy check

Use or modify the appropriate tools/<impl>/dataRedundancyInformation.sh script to generate an output file that verifies your system under test meets the redundancy requirements defined in the benchmark specification.


# Accept the Data Generator (PDGF) License

The first time you try to run the benchmark you will be asked to accept the License of the synthetic data generator.


# Running the benchmark

Run the benchmark with:

```bash
docker-compose run --build benchmark
```

Note: Use cases are numbered sequentally from 1 to 10. During the benchmark execution or in the reports generated at the end of the benchmark execution, a special artifact named "use case 0" may show up. Use case 0 is not a real use case, it is used as a special name used to record the execution of certain  parts of the benchmark that perform global operations applying to all use cases and not to a particular one (e.g. Data generation).

Before running the benchmark for the first time you will have to set the following variables in the setenv.sh file:
- TPCx-AI_VALIDATION_CONFIG_FILE_PATH: This variable must be set to the absolute path of a valid configuration file to be used for the validation test (i.e. using scale factor 1). Examples of configuration files can be found under the driver/config folder.
- TPCx-AI_BENCHMARKRUN_CONFIG_FILE_PATH: This variable must be set to the absolute path of a valid configuration file to be used for the actual benchmark run (i.e. using the actual scale factor assigned to TPCx-AI_SCALE_FACTOR). Examples of configuration files can be found under the driver/config folder.
- TPCx-AI_SCALE_FACTOR: This variable controls the scale factor used to run the benchmark. Set this variable to a value greater than or equals to 1. The scale factor represents approximately the number of gigabytes that will be generated and used to run the benchmark.
- TPCxAI_ENV_TOOLS_DIR: Location of the subdirectory containing scripts to collect system configuration information and other utilities (e.g. set TPCxAI_ENV_TOOLS_DIR=${TPCx_AI_HOME_DIR}/tools/python if you'll run the single node reference implementation or set TPCxAI_ENV_TOOLS_DIR=${TPCx_AI_HOME_DIR}/tools/spark if you'll run the multi-node implementation)
- TPCxAI_SERVING_THROUGHPUT_STREAMS: Number of current streams to use in the SERVING_THROUGHPUT test

**The following settings apply only to the Spark kit**
- YARN_CONF_DIR: Path to the directory containing the YARN configuration
- PYSPARK_PYTHON: Location of the Python binary in the virtual environment used to run the DL use cases (e.g. /usr/envs/adabench_dl)

After assigning values to configuration variables in setenv.sh, a "Validation Run" or a "Full Benchmark Run" can be executed using the TPCx-AI_Validation.sh or the TPCx-AI_Benchmarkrun.sh scripts, respectively.
The scale factor when running TPCx-AI_Validation.sh. is always 1.

After the benchmark is run, some scripts that collect system configuration information will be executed. It's possible that parts of those scripts will invoke the sudo command. In those cases you may be required to enter the user's password when the sudo command is executed unless otherwise configured in the sudoers file.



# Additional details for the multi-node reference implementation

## Configuration of spark-submit parameters ##
- Set the spark configuration parameters in the configuration file you plan to use for your execution (See section on Spark configuration parameters below).
- In the workload section of the configuration file, find the "engine_executable" configuration node and add your spark-submit parameters, such as executor-cores, executor-memory, deploy-mode, etc.
- Additional parameters for the spark-submit command that you might want to configure are: spark.rpc.message.maxSize, spark.executor.extraJavaOptions, driver-java-options, spark.kryoserializer.buffer.max, and spark.executorEnv.NUMBA_CACHE_DIR. For instance, the following parameter configurations were used in some of our tests:

e.g. engine_executable: &ENGINE "spark-submit --master yarn --conf spark.submit.deployMode=client --conf spark.executor.cores=5  --conf spark.executor.memory=32g  --conf spark.executor.memoryOverhead=8g --conf spark.rpc.message.maxSize=1024 --conf spark.kryoserializer.buffer.max=1024 --conf spark.executor.extraJavaOptions='-Xss128m' --driver-java-options '-Xss128m'"
-  These configurations are not meant to be optimal values, please adjust them to optimize the execution of the benchmark in your own systems.
- Additional parameters that can be set to potentially increase the execution performance include: --num-workers and num-threads for use case 8, --num-blocks for use case 7, and --batch size, --executor_cores_horovod, --task_cpus_horovod and --learning_rate for use cases 2, 5, and 9. These parameter can be used to fine tune the execution of the use cases and depend on the amount of resources available (CPU, Memory, etc). Some parameters have restrictions outlined in Appendix C of the specification.
- For use cases 2 and 9, there is a different Spark Session being created for the Horovod training phase (after preprocessing). To control the number of horovod workers that are spawned during training, the --executor_cores_horovod and --task_cpus_horovod parameters can be used (e.g. --executor_cores_horovod 4 --task_cpus_horovod 1). As in regular (i.e. not using Horovod) Spark sessions, these parameters control the number of cores assigned to each executor and the cores used by each task, respectively. Each task will run a Horovod worker (e.g. A TensorFlow instance). The default values for these parameters is 1. For the Spark preprocessing session (before training using Horovod), the number of cores per executor and the cpus per task can be configured using the regular --executor-cores and spark.task.cpus  configuration parameters.
- For larger scale factor (>=1000) Consider increasing the following properties from the Cloudera Manager UI as well: namenode.handler.count, dfs.datanode.handler.count, and dfs.namenode.service.handler.count.

## Loading data to HDFS ##
The LOADING phase of the benchmark will execute the *tools/python/create.sh* and *tools/python/load.sh* scripts for the single-node implementation. For the multi-node implementation, the *tools/spark/create_hdfs.sh* and the *tools/spark/load_hdfs.sh* scripts will be run during the LOADING phase. These scripts can be freely modified by the user if needed as long as they keep the same intent of copying the use cases' input data to the location from where they will be read by the use cases.

## Enabling parallel data generation and loading ###

For large scale factors (e.g. >=1000) we recommend that you enable *parallel data generation and loading* since it can reduce significantly the time required to generate data and to upload it to HDFS. When this option is enabled, the DATA_GENERATION and LOADING phases are divided among all the worker nodes in the cluster. To enable parallel data generation and loading the following configuration parameters need to be set in your configuration file (e.g. default-spark.yaml):
- pdgf_node_parallel: True
- Rename anchor &HDFS to &HDFS_BAK and rename &HDFS_PARALLEL to &HDFS in your configuration file (for instance driver/config/default-spark.yaml). Anchor &HDFS controls the commands that are executed during the LOADING phase, including create, load, delete, and download. Anchor &HDFS_PARALLEL contains commands used during parallel data generation and parallel load, in order to activate it, it needs to be renamed as &HDFS.
- Run the *tools/enable_parallel_datagen.sh* script which will copy all necessary files and tools to the worker nodes to enable parallel data generation and loading.
- For scale factors greater than 100 it's advisable that you increase the limit for the number of open files from the default to 16k or a larger value. The* /etc/security/limits.conf* file typically contains this configuration and might need your systems to be rebooted after changing it.

## Configuration Hints ##
The following hints are provided just as reference and to give an initial idea of reasonable configuration values that you can use to run the benchmark. These configuration values are not mandatory and you'll most likely need to fine-tune them to obtain optimal results.

**Hint 1: Spark-submit configuration parameters for non-DL**

You can use the following steps to determine an initial set of good parameter values (not necessarily optimal).

Set the memory available for containers to node managers in CDH (configuration property: yarn.nodemanager.resource.memory-mb) to <total memory per node>-20g.
Assuming all your cluster nodes have the same characteristics:
 * set spark.executor.cores to 5
 * The executor memory can be assigned based on each node's  "hw thread count" and "memory availability":
        set spark.executor.memory to "(<node memory> - 20g) / (FLOOR(<hw thread count> - 2)/5)
 * set spark.executor.memoryOverhead to 8g

**Hint 2: Spark-submit configuration parameters for DL**

You can use the following steps to determine an initial set of good parameter values (not necessarily optimal).
 * For DL use cases (uc2, uc5 and uc9) set spark.executor.cores to ~80% the number of cores in your system, spark.executor.memory to approximately 80% of yarn.nodemanager.resource.memory-mb  and set the parameter num-executors to the number of nodes in your cluster. This will result in running one executor per node. The number of tensorflow tasks that will run in each executor is controlled by the --executor_cores_horovod and --task_cpus_horovod parameters (Details are presented above. Each Spark task will run a tensorflow instance). You can also set spark.executor.memoryOverhead to 10% of executor-memory.


# Results
Once the benchmark finishes executing, it prints timing information and the performance metric.

Log files containing runtime and output from running the benchmark are stored in the log/history subdirectory using the following format:  tpcxai_benchmark_(validation|run)_<date>_<time>. The folder will also be compressed in a .zip file with the same name of the subdirectory. In addition, execution results are also being stored in a SQLite DB located in log/tpcxai.db.

Finally, two files named *report.txt* and *report.html* will be generated containing the final benchmark metric: *AIUCpm@SF*; and its composing elements: *Load Time (TLD)*, *Power Training Test (TPTT)*, *Power Serving Tests (TPST1, TPST2, and TPST)*, and *Throughput Test (TTT)*. The details of the metric computation can be found in the benchmark specification document available in the downloads section of the website. The report files will also contain information about the duration of the benchmark and the individual times spent in each phase of the benchmark, and

# Notes on Scale Factors (SF)

We have successfully tested this kit with the following scale factors:

**Single node (Python):** 1, 3 10, 30

**Multi-node (Spark):** 3, 10, 30, 100, 1000, and 3000


# Troubleshooting

* A known issue may arise during the Serving phase of Use Case 1 making it fail due to many threads being executed. In that case you can export the OMP_NUM_THREADS environment value with an appropriate value for your setup. You may want to start with a small value (e.g. 1) and gradually increase until you see the error show up again.

* The single node (Python) implementation of Use Case 8 may fail for scale factors larger than 20 due to a limitation on Pandas raising an "Unstacked DataFrame is too big" exception. The issue is well-documented on https://github.com/pandas-dev/pandas/pull/34827. for engineering experimentation, the following patch may allow test runs to complete: Edit "pandas/core/reshape/reshape.py" changing line "num_cells = np.multiply(num_rows, num_columns, dtype=np.int32)" to use dtype=np.int64 instead of dtype=np.int32. This allows Use Case 8 to finish execution. However, users are reminded that for formal publication pandas is part of the SUT and any changes would need to be supported by the support channel for python/pandas.

* As new versions of the packages needed to run the benchmark are released, some incompatibilities may show up. The .yml files included in the tools/python and tools/spark subdirectories list versions of packages that should work based on our tests. Users can use different versions of the packages as long as they comply with the software requirements (e.g. versioning, support, pricing, etc.) described in the specification document and the minimum versions listed in Appendix F of the specficiation document. The .yml files should be used as the primary reference of versions that work. However, for convenience some versions as listed here:

  Single node:
  | Package           | Version |
  |-------------------|---------|
  | python            | 3.7.12  |
  | setuptools        | 58      |
  | pandas            | 1.2.4   |
  | scikitlearn       | 1.0.2   |
  | xgboost           | 1.5.0   |
  | numpy             | 1.19.2  |
  | nose              | 1.3.7   |
  | scipy             | 1.7.3   |
  | statsmodels       | 0.12.2  |
  | patsy             | 0.5.2   |
  | tqdm              | 4.62    |
  | keras             | 2.3.1   |
  | tensorflow        | 2.1     |
  | joblib            | 1.1.0   |
  | pyyaml            | 6       |
  | matplotlib        | 3.5.0   |
  | jinja2            | 3.0.2   |
  | pycryptodome      | 3.12    |
  | dlib              | 19.2    |
  | pmdarima          | 1.8.2   |
  | scikitsurprise    | 1.1     |
  | librosa           | 0.8.1   |
  | imbalancedlearn   | 0.9.0   |
  | python-benedict   | 0.24    |
  | tensorflow-addons | 0.15.0  |
  | opencv            | 4.5     |

  Multi-node:
  | Package             | Version |
  |---------------------|---------|
  | gcc_linux64         | 9.3     |
  | gxx_linux64         | 9.3     |
  | openmpi-mpicc       | 4       |
  | numpy               | 1.19.2  |
  | tensorflow          | 2.2     |
  | pyspark             | 2.4     |
  | h5py                | 2.10.0  |
  | tqdm                | 4.62    |
  | joblib              | 1.1.0   |
  | pyarrow             | 3       |
  | pyopencv            | 4.5     |
  | librosa             | 0.81    |
  | dlib                | 19.2    |
  | horovod[tensorflow] | 0.19    |
  | petastorm           | 0.9.8   |
  | tensorflow-addons   | 0.10.0  |



## Filing bugs
* Please use the following URL to submit any comments, requests or bugs: https://tpcorg.fogbugz.com/default.asp?pg=pgPublicEdit&ixProject=32

## **Other options (advanced) **

You can also use the tpcxai.sh script to run certain specific parts of the benchmark. Running only certain parts of the benchmark will result in an invalid benchmark run that does not comply with the requirements and rules made by the TPC. Run "bin/tpcxai.sh -h" for a list of parameters that can be used to run the benchmark. For instance, the command to run use cases 1 and 3 (uc01 and uc03) is as follows (other parameters that can be used are explained below):

./bin/tpcxai.sh -c /opt/tpcx-ai/driver/config/default-spark.yaml -uc 1 3 -sf <sf> [-v]

- c: the configuration file to be used: For spark execution use the absolute path to <TPCx-AI_HOME>/driver/config/default-spark.yaml. For python execution use the absolute path to <TPCx-AI_HOME>/driver/config/default.yaml
- uc: The "ID"(i.e. currently a number between 1 and 10) of the use case to be executed (omit this parameter if you want to run them all)
- sf: Refers to the scale factor used for the data generation process. It's a positive integer that indicates the amount of data used to run the workload (use case). See the "Scale Factors" section below for additional comments.
- v: Verbose mode
