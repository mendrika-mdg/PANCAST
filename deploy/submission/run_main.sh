#!/bin/bash
HOSTS='sci-vm-03.jasmin.ac.uk'
RUN_HOST="$(readlink -f $(dirname ${0}))/run_host.sh"
JOB="$(readlink -f $(dirname ${0}))/run_live.sh"

for HOST in ${HOSTS}
do
  ping -c 2 -t 10 ${HOST} >& /dev/null
  if [ "${?}" = "0" ] ; then
    REMOTE_HOST=${HOST}
    break
  fi
done

if [ -z "${REMOTE_HOST}" ] ; then
  echo "No remote host available. Exiting"
  exit
fi

echo "Runninng:"
echo "  ${JOB}"
echo "On:"
echo "  ${REMOTE_HOST}"

${RUN_HOST} ${REMOTE_HOST} ${JOB} >& ${JOB}.out

