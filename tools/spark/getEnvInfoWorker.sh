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

echo "##############################"
echo "#          Hardware          #"
echo "##############################"

if type dmidecode > /dev/null 2>&1
then
  echo -e "\n##### dmidecode #####\n"
  if [ "$UID" -eq 0 ]
  then
    dmidecode
  else
    sudo dmidecode
  fi
fi
echo -e "\n##### /proc/cpuinfo #####\n" >> "$ENV_INFO_FILE" 2>&1
cat /proc/cpuinfo
echo -e "\n##### /proc/meminfo #####\n"
cat /proc/meminfo
if type lscpu > /dev/null 2>&1
then
  echo -e "\n##### lscpu #####\n"
  lscpu
fi
if type lspci > /dev/null 2>&1
then
  echo -e "\n##### lspci #####\n"
  lspci
fi
if type lsblk > /dev/null 2>&1
then
  echo -e "\n##### lsblk #####\n"
  lsblk
fi
if type mount > /dev/null 2>&1
then
  echo -e "\n##### mounted disks #####\n"
  mount
fi
if type ifconfig > /dev/null 2>&1
then
  echo -e "\n##### ifconfig #####\n"
  ifconfig
else
  if type ip > /dev/null 2>&1
  then
    echo -e "\n##### ip #####\n"
    ip addr list
  fi
fi
if type iptables-save > /dev/null 2>&1
then
  echo -e "\n##### iptables #####\n"
  if [ "$UID" -eq 0 ]
  then
    iptables-save
  else
    sudo iptables-save
  fi
fi

echo "##############################"
echo "#          Software          #"
echo "##############################"

echo -e "\n##### linux release #####\n"
cat /etc/*release
echo -e "\n##### kernel release #####\n"
uname -a
echo -e "\n##### date #####\n"
date
echo -e "\n##### Spark version #####\n"
spark-submit --version 2>&1 | cat
echo -e "\n##### hadoop version #####\n"
hadoop version
echo -e "\n##### hadoop classpath #####\n"
hadoop classpath
echo -e "\n##### java version #####\n"
java -version
echo -e "\n##### environment #####\n"
set
if type rpm > /dev/null 2>&1
then
  echo -e "\n##### installed packages (from rpm) #####\n"
  rpm -qa
fi
if type dpkg > /dev/null 2>&1
then
  echo -e "\n##### installed packages (from dpkg) #####\n"
  dpkg -l
fi
