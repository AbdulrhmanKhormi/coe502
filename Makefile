# Directory
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Compilation Configuratoin
CC:=gcc
IFLAGS:=-lm
CFLAGS:=-mavx -mavx2 -O3

# File and directory names
BUILD_DIR := $(ROOT_DIR)/build
SRC_DIR := $(ROOT_DIR)/src

# Get all possible benchmarks
BENCHMARKS := $(notdir $(shell dirname $(shell find $(SRC_DIR)/ -mindepth 2 -maxdepth 2 -name "Makefile.mk")))

# Output
BINS_BM  := $(addprefix $(BUILD_DIR)/,$(BENCHMARKS))
CLEAN_BM := $(addprefix clean_,$(BENCHMARKS))

# Default
all: $(BINS_BM)

# Build directory
$(BUILD_DIR):
	mkdir -p $@

# Clean
clean: $(CLEAN_BM)
	rm -rf $(BUILD_DIR)

# Template
include template.mk

# All benchmarks/applications
-include $(SRC_DIR)/Makefile.mk

.PHONY: clean all

#  {test, dev, small, medium, large, native}
DATA:= -d native
nThreads:= 24

run-scalar:
	./build/blackscholes -i scalar $(DATA)

run-vec:
	./build/blackscholes -i vec $(DATA)

run-para:
	./build/blackscholes -i para $(DATA) -n $(nThreads)

run-vec-para:
	./build/blackscholes -i vec-para $(DATA) -n $(nThreads)

run-all: run-scalar run-para run-vec run-vec-para

run-all-datasets:
	./build/blackscholes -i scalar -d dev
	./build/blackscholes -i vec -d dev
	./build/blackscholes -i para -d dev -n $(nThreads)

	./build/blackscholes -i scalar -d small
	./build/blackscholes -i vec -d small
	./build/blackscholes -i para -d small -n $(nThreads)

	./build/blackscholes -i scalar -d medium
	./build/blackscholes -i vec -d medium
	./build/blackscholes -i para -d medium -n $(nThreads)

	./build/blackscholes -i scalar -d large
	./build/blackscholes -i vec -d large
	./build/blackscholes -i para -d large -n $(nThreads)

	./build/blackscholes -i scalar -d native
	./build/blackscholes -i vec -d native
	./build/blackscholes -i para -d native -n $(nThreads)

save-results:
	make run-all-datasets | grep 'Runtimes\|Running\|dataset' > result.txt

