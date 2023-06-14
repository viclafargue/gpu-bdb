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
docker-compose run --build benchmark -uc 4
```

*Note, currently only UC4 is tested and validated to work.*

Note: Use cases are numbered sequentally from 1 to 10. During the benchmark execution or in the reports generated at the end of the benchmark execution, a special artifact named "use case 0" may show up. Use case 0 is not a real use case, it is used as a special name used to record the execution of certain  parts of the benchmark that perform global operations applying to all use cases and not to a particular one (e.g. Data generation).
