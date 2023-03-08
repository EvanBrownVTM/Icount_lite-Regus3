#!/bin/bash

version=v1.2

echo $'Purging current software..\n'
sudo rm -r ~/Desktop/Icount_lite 

echo $'Fetching software version.. $version\n'
cd ~/Desktop

curl --header "PRIVATE-TOKEN: glpat-DeSKjfLDqTFsx_zXcqpZ" -L "https://gitlab.com/api/v4/projects/35394900/repository/archive.zip?sha=$version" -o archive.zip

echo $'\nExtracting files..'
dir_name=$(unzip -qql archive.zip | head -n1 | tr -s ' ' | cut -d' ' -f5-)
echo $'\nFetched dir: $dir_name'
unzip archive.zip

echo $'\nRenaming fetched directory..'
mv $dir_name Icount_lite

echo $'\nFile clean up'
rm -r archive.zip


