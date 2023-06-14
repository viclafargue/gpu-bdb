#!/bin/sh

FN="data-gen/Constants.properties"

if [ -f "${FN}" ] && grep -q "IS_EULA_ACCEPTED=true" "${FN}"; then
    echo "EULA is already accepted."
    exit 0
else
    echo "By using this software you must first agree to TPC's terms of use."
    exit 1
fi
