#!/usr/bin/env bash

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


ENV_INFO_DIR_RELATIVE="envInfo-$HOSTNAME"
ENV_INFO_DIR="${TPCx_AI_HOME_DIR}/logs/$ENV_INFO_DIR_RELATIVE"
[ -d "$ENV_INFO_DIR" ] && rm -rf "$ENV_INFO_DIR"
mkdir "$ENV_INFO_DIR"
ENV_INFO_FILE="$ENV_INFO_DIR/envInfo.log"
> "$ENV_INFO_FILE"
if [ -w "$ENV_INFO_FILE" ]
then
  echo "##############################" >> "$ENV_INFO_FILE" 2>&1
  echo "#          Hardware          #" >> "$ENV_INFO_FILE" 2>&1
  echo "##############################" >> "$ENV_INFO_FILE" 2>&1

  if type dmidecode > /dev/null 2>&1
  then
    echo -e "\n##### dmidecode #####\n" >> "$ENV_INFO_FILE" 2>&1
    if [ "$UID" -eq 0 ]
    then
      dmidecode >> "$ENV_INFO_FILE" 2>&1
    else
      sudo dmidecode >> "$ENV_INFO_FILE" 2>&1
    fi
  fi
  echo -e "\n##### /proc/cpuinfo #####\n" >> "$ENV_INFO_FILE" 2>&1
  cat /proc/cpuinfo >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n##### /proc/meminfo #####\n" >> "$ENV_INFO_FILE" 2>&1
  cat /proc/meminfo >> "$ENV_INFO_FILE" 2>&1
  if type lscpu > /dev/null 2>&1
  then
    echo -e "\n##### lscpu #####\n" >> "$ENV_INFO_FILE" 2>&1
    lscpu >> "$ENV_INFO_FILE" 2>&1
  fi
  if type lspci > /dev/null 2>&1
  then
    echo -e "\n##### lspci #####\n" >> "$ENV_INFO_FILE" 2>&1
    lspci >> "$ENV_INFO_FILE" 2>&1
  fi
  if type lsblk > /dev/null 2>&1
  then
    echo -e "\n##### lsblk #####\n" >> "$ENV_INFO_FILE" 2>&1
    lsblk >> "$ENV_INFO_FILE" 2>&1
  fi
  if type mount > /dev/null 2>&1
  then
    echo -e "\n##### mounted disks #####\n" >> "$ENV_INFO_FILE" 2>&1
    mount >> "$ENV_INFO_FILE" 2>&1
  fi
  if type ifconfig > /dev/null 2>&1
  then
    echo -e "\n##### ifconfig #####\n" >> "$ENV_INFO_FILE" 2>&1
    ifconfig >> "$ENV_INFO_FILE" 2>&1
  else
    if type ip > /dev/null 2>&1
    then
      echo -e "\n##### ip #####\n" >> "$ENV_INFO_FILE" 2>&1
      ip addr list >> "$ENV_INFO_FILE" 2>&1
    fi
  fi
  if type iptables-save > /dev/null 2>&1
  then
    echo -e "\n##### iptables #####\n" >> "$ENV_INFO_FILE" 2>&1
    if [ "$UID" -eq 0 ]
    then
      iptables-save >> "$ENV_INFO_FILE" 2>&1
    else
      sudo iptables-save >> "$ENV_INFO_FILE" 2>&1
    fi
  fi

  echo "##############################" >> "$ENV_INFO_FILE" 2>&1
  echo "#          Software          #" >> "$ENV_INFO_FILE" 2>&1
  echo "##############################" >> "$ENV_INFO_FILE" 2>&1

  echo -e "\n##### linux release #####\n" >> "$ENV_INFO_FILE" 2>&1
  cat /etc/*release >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n##### kernel release #####\n" >> "$ENV_INFO_FILE" 2>&1
  uname -a >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n##### date #####\n" >> "$ENV_INFO_FILE" 2>&1
  date >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n##### java version #####\n" >> "$ENV_INFO_FILE" 2>&1
  java -version >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n##### Linux environment #####\n" >> "$ENV_INFO_FILE" 2>&1
  set >> "$ENV_INFO_FILE" 2>&1
  if type rpm > /dev/null 2>&1
  then
    echo -e "\n##### installed packages (from rpm) #####\n" >> "$ENV_INFO_FILE" 2>&1
    rpm -qa >> "$ENV_INFO_FILE" 2>&1
  fi
  if type dpkg > /dev/null 2>&1
  then
    echo -e "\n##### installed packages (from dpkg) #####\n" >> "$ENV_INFO_FILE" 2>&1
    dpkg -l >> "$ENV_INFO_FILE" 2>&1
  fi
  echo -e "\n##### Python environment #####\n" >> "$ENV_INFO_FILE" 2>&1
  ${TPCx_AI_HOME_DIR}/lib/python-venv/bin/python --version >> "$ENV_INFO_FILE" 2>&1
  ${TPCx_AI_HOME_DIR}/lib/python-venv/bin/pip list >> "$ENV_INFO_FILE" 2>&1
  echo -e "\n\n" >> "$ENV_INFO_FILE" 2>&1
  echo -e "\nPython clock Info - Perf_counter \n" >> "$ENV_INFO_FILE" 2>&1
  ${TPCx_AI_HOME_DIR}/lib/python-venv/bin/python -c "import time; print(time.get_clock_info('perf_counter'))" >> "$ENV_INFO_FILE" 2>&1
  echo -e "\nPython clock Info - time \n" >> "$ENV_INFO_FILE" 2>&1
  ${TPCx_AI_HOME_DIR}/lib/python-venv/bin/python -c "import time; print(time.get_clock_info('time'))" >> "$ENV_INFO_FILE" 2>&1

  cd ${TPCx_AI_HOME_DIR}/logs
  zip -r "$ENV_INFO_DIR_RELATIVE.zip" "$ENV_INFO_DIR_RELATIVE"
else
  echo "environment information could not be written to $ENV_INFO_FILE"
  exit 1
fi
