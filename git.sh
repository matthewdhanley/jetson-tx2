#!/bin/bash
echo 'This will overwrite any local changes you have made.'
echo 'Continue? (y/n)'
read answer
if [ "$answer" -eq "y" ]
then
	git fetch --all
	git reset --hard origin/master
fi
