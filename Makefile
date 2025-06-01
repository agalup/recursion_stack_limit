
all:
	nvcc -std=c++17 -rdc=true -O2 --gpu-architecture=sm_80 bad_usage.cu -o bad_usage -lcuda
