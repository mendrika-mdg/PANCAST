#!/bin/bash

# Directory where temporaryy ssh keys will be stored:
KEY_DIR="${HOME}/.k"
# Get uuid for temp file naming:
UUID=$(cat /proc/sys/kernel/random/uuid)
# Path to temp ssh key:
KEY_FILE="${KEY_DIR}/key__${UUID}"

# Exit function:
function x_it() {
  # Remove temporary files:
  \rm -f ${KEY_FILE} ${KEY_FILE}.pub
  sed -i "/${UUID}@jasmin/d" ${HOME}/.ssh/authorized_keys
  # Exit:
  exit
}

# Catch these signals for exiting:
trap x_it SIGINT
trap x_it SIGKILL
trap x_it TERM

# Create key directory if it does not exist:
if [ ! -d "${KEY_DIR}" ] ; then
  mkdir -p ${KEY_DIR}
  chmod 700 ${KEY_DIR}
fi

# Remove any temporary key files older than one day:
find ${KEY_DIR} \
  -mindepth 1 -maxdepth 1 -type f -name 'key__*' -mtime +1 \
  -exec \rm '{}' \;

# Create the key:
ssh-keygen -N '' -t rsa -b 2048 -C "${UUID}@jasmin" -f ${KEY_FILE} >& /dev/null
# Add key to authorized_keys:
cat ${KEY_FILE}.pub >> ${HOME}/.ssh/authorized_keys
# Connect using temporary key:
ssh -i ${KEY_FILE} -q -t "${@}"
# Exit:
x_it
