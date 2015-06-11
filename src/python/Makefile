.PHONY: swig python all clean

# Sick makefile for all your python cvxcanon needs

swig:
	swig -outcurrentdir -c++ -python -I../ CVXcanon.i
	$(MAKE) -C ../../external/pogs/src
	g++ -DNDEBUG -g -shared -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 \
	-L /Users/stevend2/anaconda/envs/abs_ops/lib/ \
	-l python2.7 \
	-I/Users/stevend2/anaconda/envs/abs_ops/lib/python2.7/site-packages/numpy/core/include \
	-I../ -I../../external/pogs/src/include/ \
	-I/Users/stevend2/anaconda/envs/abs_ops/include/python2.7 \
	../LinOpOperations.cpp ../CVXcanon.cpp CVXcanon_wrap.cxx \
	../../external/pogs/src/build/pogs.a \
	-framework Accelerate \
	-o _CVXcanon.so
	# g++ -DNDEBUG -g -shared -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 \
	# `python-config --include --ldflags` \
	# -I/Users/stevend2/anaconda/lib/python2.7/site-packages/numpy/core/include \
	# -I../ -I../../external/pogs/src/include/ \
	# -I/Users/stevend2/anaconda/include/python2.7 \
	# ../LinOpOperations.cpp ../CVXcanon.cpp CVXcanon_wrap.cxx \
	# ../../external/pogs/src/build/pogs.a \
	# -framework Accelerate \
	# -o _CVXcanon.so
	# cp CVXcanon.py _CVXcanon.so ../../tests/python

python:
	cd ../../include/cvxpy; sudo python setup.py install;

clean:
	$(RM) *.*o
	$(RM) *.pyc
	$(RM) CVXcanon.py
	$(RM) *_wrap.cxx
	# $(RM) ../../tests/python/CVXcanon.py
	# $(RM) ../../tests/python/_CVXcanon.so
	$(MAKE) clean -C ../../external/pogs/src

all:
	clean
	swig
	python