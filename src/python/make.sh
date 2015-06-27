cd ../../include/pogs_fork/src/
make gpu
cd ../../scs
make
cd ../../
cd src/python

swig -python -c++ -I../ -outcurrentdir -o FAO_DAG_wrap.cu FAO_DAG.i

nvcc -Xcompiler -fno-strict-aliasing,-fno-common,-dynamic,-fwrapv,-Wall,-Wstrict-prototypes \
-I/usr/local/include -I/usr/local/opt/sqlite/include \
-DNDEBUG -g -O3  -DLAPACK_LIB_FOUND \
-I/Users/stevend2/anaconda/envs/abs_ops/lib/python2.7/site-packages/numpy/core/include \
-I../ -I../../include/ -I../../include/scs/include/ \
-I../../include/pogs_fork/src/include/ \
-I/Users/stevend2/anaconda/envs/abs_ops/include/python2.7 \
-c FAO_DAG_wrap.cu \
-o FAO_DAG_wrap.o \
-std=c++11 -DFAO_GPU

g++ -bundle -undefined dynamic_lookup \
-L/Users/stevend2/anaconda/envs/abs_ops/lib \
-arch x86_64 -L/usr/local/include \
-I/usr/local/include \
FAO_DAG_wrap.o ../../include/pogs_fork/src/build/pogs.a \
-L ../../include/scs/out -lscsindir \
-L/usr/local/cuda/lib -L/usr/local/lib \
-lcudart -lcublas -lcusparse \
-L/usr/lib -L/usr/lib \
-L/Users/stevend2/anaconda/envs/abs_ops/lib \
-lfftw3 -lfftw3f -lfftw3l -lfftw3_threads \
-lfftw3f_threads -lfftw3l_threads \
-o _FAO_DAG.so
# echo ' ' >> FAO_DAG.i
# python setup.py install
# python ../../test/test_pogs_mat_free.py
