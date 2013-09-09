CUDA?=/usr/local/cuda
CXX=g++ -DLINUX -w
LIB=lib
ifeq ($(shell uname), Linux)
  ifeq ($(shell uname -m), x86_64)
	LIB=lib64
  endif
endif
PWD?=$(shell pwd)

INCLUDES +=\
	-I$(CUDA)/include  \
	-I/usr/include

LIBRARIES +=\
	-L$(CUDA)/$(LIB) -lcuda -lcudart -lnvToolsExt

NVCC=$(CUDA)/bin/nvcc
CUDA_OPTIMISE=-O3
NVCCFLAGS += -ccbin $(CXX) $(ARCH_FLAGS) $(CUDA_DEBUG) $(CUDA_OPTIMISE)\
	-gencode=arch=compute_20,code=sm_20 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_35,code=sm_35 \
	-lineinfo --machine 64 \
	-Xcompiler -fPIC

all: bin/cudaBenchmark bin/transpose bin/reduction Makefile

bin/cudaBenchmark: src/cudaBenchmark.cu
	@$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@
	@echo creating $@ from $<

bin/transpose: src/transpose.cu
	@$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@
	@echo creating $@ from $<

bin/reduction: src/reduction.cu
	@$(NVCC) $(NVCCFLAGS) $(LIBRARIES) $(INCLUDES) $(DEFINES)  $< -o $@
	@echo creating $@ from $<

.PHONY: clean
clean:
	@echo "Cleaning..."
	@-rm bin/*
	@echo "Complete!"
