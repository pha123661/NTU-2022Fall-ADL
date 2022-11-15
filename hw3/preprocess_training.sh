FILE_PATH="./data/"
for NAME in public train sample_test
do
    python jsonl2json.py -i="$FILE_PATH$NAME.jsonl" -o="$FILE_PATH$NAME.json"
done