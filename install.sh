#!/bin/bash

BUILD_DIR=$1
DST_DIR=$2

cp $BUILD_DIR/libarm_compute_core-static.a $DST_DIR
cp $BUILD_DIR/libarm_compute_graph-static.a $DST_DIR
cp $BUILD_DIR/libarm_compute-static.a $DST_DIR
