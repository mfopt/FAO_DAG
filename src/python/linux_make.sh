cd ../../include/pogs_fork/src/
make gpu
cd ../../scs
make
cd ../../
cd src/python

swig -python -c++ -I../ -outcurrentdir -o FAO_DAG_wrap.cu FAO_DAG.i

nvcc -Xcompiler -fPIC,-fno-strict-aliasing,-fno-common,-dynamic,-fwrapv,-Wall,-Wstrict-prototypes \
-I/usr/local/include -I/usr/local/opt/sqlite/include \
-DNDEBUG -g -O3  -DLAPACK_LIB_FOUND \
-I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
-I../ -I../../include/ -I../../include/scs/include/ \
-I../../include/pogs_fork/src/include/ \
-I/usr/include/python2.7 \
-c FAO_DAG_wrap.cu \
-o FAO_DAG_wrap.o \
-std=c++11 -DFAO_GPU

g++ -shared -L/usr/local/lib \
-L/usr/local/include \
-I/usr/local/include \
FAO_DAG_wrap.o ../../include/pogs_fork/src/build/pogs.a \
-L ../../include/scs/out -lscsindir \
-L/usr/local/cuda/lib64 -L/usr/local/lib \
-lcuda -lcudart -lcublas -lcusparse -lcufftw \
-L/usr/lib -L/usr/lib \
-L/usr/lib \
-lblas -llapack -lgfortran \
-lfftw3 -lfftw3f -lfftw3l -lfftw3_threads \
-lfftw3f_threads -lfftw3l_threads -lpython2.7 \
-o _FAO_DAG.so
# echo ' ' >> FAO_DAG.i
# python setup.py install
# python ../../test/test_pogs_mat_free.py
