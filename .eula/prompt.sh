#!/bin/sh

.eula/check.sh

if [ $? -eq 0 ]; then
    exit 0
fi

# read -r -p "By using this software you must first agree to TPC's terms of use. press [ENTER] to show them"
echo "Press [ENTER] to show them" && read dummy_var
cat EULA.txt

# Prompt the user to accept the EULA.
echo
echo "If you have read and agree to this terms of use, please type (uppercase!):  YES  and press [ENTER]"
read response

# Check if the entered word matches the expected word
if [ "$response" = "YES" ]; then
    echo "#" > data-gen/Constants.properties
    echo "#$(date --utc)" >> data-gen/Constants.properties
    echo "IS_EULA_ACCEPTED=true" >> data-gen/Constants.properties
else
    echo "You must accept the EULA to use this software. Exiting..."
    exit 1
fi
