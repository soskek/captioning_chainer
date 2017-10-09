echo "\n\n\t# DOWNLOAD\n\n"
sh prepare_scripts/download.sh

echo "\n\n\t# UNZIP\n\n"
sh prepare_scripts/unzip.sh

echo "\n\n\t# COUNT and MAKE VOCABULARY\n\n"
sh prepare_scripts/vocab.sh
