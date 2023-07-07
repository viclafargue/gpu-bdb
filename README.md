# The RAPIDS GPU Express Benchmark AI

This benchmark repository is derived and from the TPC Express Benchmark AI (TPCx-AI) benchmark. It was modified to be used within the context of RAPIDS software, results cannot be directly compared to those obtained with the unmodified TPCx-AI benchmark reference implementation.
TPC Express Benchmark AI (TPCx-AI) is an AI benchmark that models several aspects of an AI or machine learning data science pipeline using a diverse dataset that includes images, text and audio. The benchmark defines and provides a means to evaluate the System Under Test (SUT) performance as a general-purpose data science system.

**THIS REPOSITORY IS IN A VERY EARLY DEVELOPMENT STAGE!**

## Clone this repository

This repository uses **git-lfs** to track large files instead of directly committing them to the repository.
Please make sure to install **git-lfs** on your system, e.g., by following [these instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), *before* cloning this repository.

You can test whether **git-lfs** is already installed by running

```bash
git lfs install
```

*Note to contributors:* Please track large files (> ~1MB) with **git-lfs** instead of committing them directly to the repository.

## Accept the Data Generator (PDGF) License

The first time you try to run the benchmark you will be asked to accept the License of the synthetic data generator.
You can do so by executing

```bash
docker-compose run accept-eula
```

## Running the benchmark

Run the benchmark, e.g., for UC4, with:

```bash
docker-compose run benchmark -uc 4
```

*Note, currently only UC4 is tested and validated to work.*

Note: Use cases are numbered sequentally from 1 to 10. During the benchmark execution or in the reports generated at the end of the benchmark execution, a special artifact named "use case 0" may show up. Use case 0 is not a real use case, it is used as a special name used to record the execution of certain  parts of the benchmark that perform global operations applying to all use cases and not to a particular one (e.g. Data generation).

## Running the benchmark with Legate

To run the benchmark with Legate, we need to specify that we want to use Legate both for building the image and running the container.

Building the container for Legate requires access to the private https://github.com/rapidsai/raft-legate repository.
To enable the build, first create a GitHub personal access token with read permissions to the raft-legate repository and then configure the build environment as follows:

1. Clone this (the gpu-xb-ai) repository.
3. Within the just cloned repository, create a file called `.secrets/github_token` that contains the earlier created PAT.

Then build the Legate container with the following command:

```bash
GITHUB_USER=<github_user> docker-compose -f compose.yaml -f compose.legate.yaml build
```
where you replace `<github_user>` with your actual GitHub username.

Run the benchmark, e.g., w/ UC4:
```bash
docker-compose -f compose.yaml -f compose.legate.yaml run --rm -e LEGATE_CONFIG='--gpus=8' benchmark -uc 4
```

Control the execution environment by setting the `LEGATE_CONFIG` environment variable within the container.
