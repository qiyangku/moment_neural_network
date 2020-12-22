CU_APPS= fast_dawson
IF_APPS= fast_dawson_if
LIBRARIES +=  -lcudart

# Common includes and paths for CUDA
INCLUDES  := -I$(HOME)/pybind11-2.6.0/include # 编译结果很干净 比2.2.0或huarui 干净
#-I/home1/wangdai/pybind11/include #-I$(CUDA_HOME)/include -I$(CONDA_HOME)/include
# INCLUDES  := -I$(HOME)/pybind11_huarui/include
# INCLUDES  := -I$(HOME)/pybind11-2.6.1/include
# INCLUDES  := -I$(HOME)/pybind11-2.22.0/include

all: ${CU_APPS} ${CU_APPS}

# %.o: %.cu
${CU_APPS}.o: ${CU_APPS}.cu
#	nvcc -O2 -arch=sm_61 -o $@ $< -lcudadevrt --relocatable-device-code true
	nvcc -std=c++11 -Xcompiler -fPIC  -arch=sm_61 $(INCLUDES) `pkg-config --cflags python3` -o $@ -c $<

${IF_APPS}.o: ${IF_APPS}.cpp
	g++  -std=c++11  -fPIC  $(INCLUDES) `pkg-config --cflags python3` -o ${IF_APPS}.o -c ${IF_APPS}.cpp
# g++ is also OK

#%:%.cpp
#	g++  -std=c++11  -fPIC  $(INCLUDES) `pkg-config --cflags python3` -o $@ $<

${CU_APPS}: ${CU_APPS}.o ${IF_APPS}.o
#	gcc -o ${CU_APPS}`python3-config --extension-suffix` -shared -fPIC -std=c++11 $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)
	g++ -o ${CU_APPS}.so -shared -fPIC -std=c++11 $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)
# $< -lcudadevrt --relocatable-device-code true



#-o $@`python3-config --extension-suffix` $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)

# c++ -O3 -Wall -shared -std=c++11 -fPIC -I/home1/wangdai/anaconda3/include/python3.7m -I/home1/wangdai/pybind11-2.6.0/include  example.cpp -o example`python3-config --extension-suffix`

#  pybind11 example build command
# c++ -O3 -Wall -shared -std=c++11 -fPIC -I/home1/wangdai/anaconda3/include/python3.7m -I/home1/wangdai/pybind11/include  example.cpp -o example`python3-config --extension-suffix`


%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS}.o  ${CU_APPS}.so ${C_APPS}.o ${IF_APPS}.o
