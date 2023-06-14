#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them
# is governed by the express license under which they were provided to you ("License"). Unless the
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.
#
#


import horovod.spark
import horovod.tensorflow.keras as hvd
from pyspark.sql import SparkSession

session = (
    SparkSession.Builder().appName("horovod test").master("local[*]").getOrCreate()
)


def fn(magic_number):
    hvd.init()
    print(
        "Hello, rank = %d, local_rank = %d, size = %d, local_size = %d, magic_number = %d"
        % (hvd.rank(), hvd.local_rank(), hvd.size(), hvd.local_size(), magic_number)
    )
    return hvd.rank()


horovod.spark.run(fn, args=(42,))