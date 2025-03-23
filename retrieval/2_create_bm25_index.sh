#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/combined_data \
  --index data/combined_data_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw