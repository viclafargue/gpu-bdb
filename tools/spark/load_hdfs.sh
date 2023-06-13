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

# Exit if any command fails
set -e

#Destination folder and space-separated list of files/directories to load
DESTINATION=$1
FILES_TO_LOAD=${@:2}

#Loop through list of files/folders to upload
for FILE in $FILES_TO_LOAD; do

   # If it's a regular file copy it to hdfs, else if its a directory create a
   # sequence file and copy the sequence file to hdfs
   if [ -f $FILE ]; then
      echo "Loading file $FILE to $DESTINATION"
      hdfs dfs -copyFromLocal -f $FILE $DESTINATION
   elif [ -d $FILE ]; then
      echo "Creating sequence file ${FILE}.seq from directory $FILE"
      bash tools/dir2seq.sh $FILE ${FILE}.seq
      if [ $? -eq 0 ]; then
         echo "Loading ${FILE}.seq to $DESTINATION"
         hdfs dfs -copyFromLocal -f ${FILE}.seq $DESTINATION
      fi
   fi
done
