#!/bin/bash

set -e
s=${1:-en}

base="cc"
#base="wiki"

echo "Downloading the ${base} ${s} model"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

if [ $base == "cc" ]; then

  src_emb_gz=data/${base}.${s}.300.vec.gz
  src_emb=data/${base}.${s}.300.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${src_emb_gz}
  fi

else

  src_emb_gz=data/${base}.${s}.vec.gz
  src_emb=data/${base}.${s}.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
    #curl -Lo data/wiki.${s}.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${s}.vec
    gzip -dkv ${src_emb_gz}
  fi

fi
