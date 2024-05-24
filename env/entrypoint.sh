#!/bin/sh
echo "The time is: $(date)"

# bash -c 'apt update;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo "$PUBLIC_KEY" >> authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity'

# # Start SSH service
# service ssh start

# Execute any other commands passed to the script
exec "$@"