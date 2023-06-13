#!/bin/bash

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
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them
# is governed by the express license under which they were provided to you ("License"). Unless the
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.
#
#


ARGS=$@

#SCRIPT=$(readlink -f "$0")
#SCRIPTPATH=$(dirname "$SCRIPT")
#TPCXAI_HOME=$(readlink -f "$SCRIPTPATH/..")

if [ -z "${TPCx_AI_HOME_DIR}" ]
   then echo "The Environment variable TPCx_AI_HOME_DIR is empty. Please edit and source the setenv.sh file before running the benchmark."
   exit 1
fi

TPCXAI_HOME=${TPCx_AI_HOME_DIR}
DRIVER_PATH="$TPCXAI_HOME/driver"

DEFAULT_CONFIG="$TPCXAI_HOME/driver/config/default.yaml"

PYTHON="$TPCXAI_HOME/lib/python-venv/bin/python"

if [[ $ARGS != *"-c"* && $ARGS != *"--config" ]]; then
  ARGS="-c $DEFAULT_CONFIG $ARGS"
fi

CMD="$PYTHON -u -m tpcxai-driver $ARGS"

# TODO fix tpcxai_home in the driver itself
# change to different working directory
cd "$DRIVER_PATH"
$CMD
