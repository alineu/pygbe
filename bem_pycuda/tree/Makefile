SWIG = swig
SWIG_OPTS = -c++ -python

CC = g++
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

#PY_VER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY_VER=3.7
#PY_INC=/home/mehdizadehrahimi.a/.conda/envs/pygbe-env-py3/include/python3.7m
PY_INC=/usr/local/include/python$(PY_VER)
#PY_LIB=/home/mehdizadehrahimi.a/.conda/pkgs/numpy-1.19.5-py37haa41c4c_1/lib/python3.7/site-packages/numpy/core/include/numpy
PY_LIB=/usr/local/lib/python$(PY_VER)/site-packages/numpy/core/include
PY_LIB_=/usr/local/lib/python$(PY_VER)/dist-packages/numpy/core/include

all: calc_mult mult dir link

calc_mult: calculateMultipoles.i
	$(SWIG) $(SWIG_OPTS) $?

mult: multipole.i
	$(SWIG) $(SWIG_OPTS) $?

dir: direct.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c calculateMultipoles.cpp calculateMultipoles_wrap.cxx -I $(PY_INC) -I $(PY_INC)m -I $(PY_LIB_) -I $(PY_LIB)
	$(CC) $(OPTS) -shared $? -o _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o
	$(CC) $(OPTS) -c multipole.cpp multipole_wrap.cxx -I $(PY_INC) -I $(PY_INC)m -I $(PY_LIB_) -I $(PY_LIB)
	$(CC) $(OPTS) -shared $? -o _multipole.so multipole.o multipole_wrap.o
	$(CC) $(OPTS) -c direct.cpp direct_wrap.cxx -I $(PY_INC) -I $(PY_INC)m -I $(PY_LIB_) -I $(PY_LIB)
	$(CC) $(OPTS) -shared $? -o _direct.so direct.o direct_wrap.o

clean:
	rm -f _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o calculateMultipoles_wrap.cxx calculateMultipoles.py *.pyc
	rm -f _multipole.so multipole.o multipole_wrap.o multipole_wrap.cxx multipole.py *.pyc
	rm -f _direct.so direct.o direct_wrap.o direct_wrap.cxx direct.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
