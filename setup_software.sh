#!/bin/bash

version=$USER


echo $'Fetching current user..\n'
echo $version
echo $'Updating name in scripts.'
sed -i '1s;^;#####DEVICE DETAILS######\n;' utils_lite/configSrc.py
sed -i '0,/user/c\user = "'$version'"' utils_lite/configSrc.py
echo $'Update finished'


echo $'\nBuilding yolo components..'
cd plugins
make
echo $'Build finished.'


