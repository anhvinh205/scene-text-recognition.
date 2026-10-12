set -eu
ZIP="${1:?usage: setup_data.sh path/to/icdar2003.zip}"
mkdir -p data
unzip -q -o "$ZIP" -d data
test -f data/SceneTrialTrain/words.xml && echo "OK: data/SceneTrialTrain/words.xml"