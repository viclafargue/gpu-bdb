# The RAPIDS GPU Express Benchmark AI

This benchmark repository is derived and from the TPC Express Benchmark AI (TPCx-AI) benchmark. It was modified to be used within the context of RAPIDS software, results cannot be directly compared to those obtained with the unmodified TPCx-AI benchmark reference implementation.
TPC Express Benchmark AI (TPCx-AI) is an AI benchmark that models several aspects of an AI or machine learning data science pipeline using a diverse dataset that includes images, text and audio. The benchmark defines and provides a means to evaluate the System Under Test (SUT) performance as a general-purpose data science system.

**THIS REPOSITORY IS IN A VERY EARLY DEVELOPMENT STAGE!**

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
