#!/bin/sh -l

TARGET=$1
IS_ALPHA=$2
IS_BETA=$3
VERSION=$4

if $IS_ALPHA; then
  SUFIX=a
elif $IS_BETA; then
  SUFIX=b
fi

VERSION=${VERSION}${SUFIX:+$SUFIX}
sed -i "1s/.*/__version__ = \"${VERSION}\"/" ${TARGET}
cat ${TARGET}
