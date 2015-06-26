.PHONY: swig python all clean

# Sick makefile for all your python cvxcanon needs

swig:
	swig -python -c++ -I../ -outcurrentdir -o FAO_DAG_wrap.cpp FAO_DAG.i

	gcc -DNDEBUG -std=c++11 -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/usr/local/include \
	-DLAPACK_LIB_FOUND -I/Users/stevend2/anaconda/envs/abs_ops/lib/python2.7/site-packages/numpy/core/include \
	-I../ -I../../include/ -I../../include/scs/include/ -I/Users/stevend2/anaconda/envs/abs_ops/include/python2.7 \
	-c FAO_DAG_wrap.cpp

	g++ -bundle -undefined dynamic_lookup \
	-L/Users/stevend2/anaconda/envs/abs_ops/lib \
	-arch x86_64 -L/usr/local/include \
	-I/usr/local/include \
	-arch x86_64 build/temp.macosx-10.5-x86_64-2.7/FAO_DAG_wrap.o \
	-L ../../include/scs/out -lscsindir \
	-L/usr/lib -L/usr/lib \
	 -L/Users/stevend2/anaconda/envs/abs_ops/lib \
	 -lfftw3 -lfftw3f -lfftw3l -lfftw3_threads \
	 -lfftw3f_threads -lfftw3l_threads \
	 -lblas -llapack -o build/lib.macosx-10.5-x86_64-2.7/_FAO_DAG.so

python:
	cd ../../include/cvxpy; sudo python setup.py install;

clean:
	rm -rf *.*o
	$(RM) *.pyc
	$(RM) FAO_DAG.py
	$(RM) *_wrap.cpp
	# $(RM) ../../tests/python/CVXcanon.py
	# $(RM) ../../tests/python/_CVXcanon.so
	# $(MAKE) clean -C ../../include/pogs/src

all:
	clean
	swig
	python
