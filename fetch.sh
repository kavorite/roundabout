curl -LO \
    https://huggingface.co/datasets/sanchit-gandhi/gtzan/blob/main/data/train-00000-of-00003-abaaa5719027ce5c.parquet \
    https://huggingface.co/datasets/sanchit-gandhi/gtzan/blob/main/data/train-00001-of-00003-40e2de07ad428882.parquet \
    https://huggingface.co/datasets/sanchit-gandhi/gtzan/blob/main/data/train-00002-of-00003-6e2eb838540a06e5.parquet \

mkdir -p gtzan
mv train-00000-of-00003-abaaa5719027ce5c.parquet gtzan/train-00000.parquet
mv train-00001-of-00003-40e2de07ad428882.parquet gtzan/train-00001.parquet
mv train-00002-of-00003-6e2eb838540a06e5.parquet gtzan/train-00002.parquet
