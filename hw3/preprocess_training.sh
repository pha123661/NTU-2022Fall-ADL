FILE_PATH="./data/"
for NAME in public train
do
    python jsonl2json.py -i="$FILE_PATH$NAME.jsonl" -o="$FILE_PATH$NAME.json"
done