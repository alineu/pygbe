SWIG = swig
SWIG_OPTS = -c++ -python

CC = g++ 
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

PY_VER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.wri$
#PY_INC=/home/mehdizadehrahimi.a/.conda/envs/pygbe-env-py3/include/python3.7m
PY_INC=/usr/local/include/python$(PY_VER)
#PY_LIB=/home/mehdizadehrahimi.a/.conda/pkgs/numpy-1.19.5-py37haa41c4c_1/lib/python3.7/site-packages/numpy/$
PY_LIB=/usr/local/lib/python$(PY_VER)/site-packages/numpy/core/include
PY_LIB_=/usr/local/lib/python$(PY_VER)/dist-packages/numpy/core/include

all: swig link

swig: semi_analyticalwrap.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c semi_analyticalwrap.cpp semi_analyticalwrap_wrap.cxx -I $(PY_INC) -I $(PY_INC)m -I $(PY_LIB) -I $(PY_LIB_)
	$(CC) $(OPTS) -shared $? -o _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o

clean:
	rm -f _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o semi_analyticalwrap_wrap.cxx semi_analyticalwrap.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
