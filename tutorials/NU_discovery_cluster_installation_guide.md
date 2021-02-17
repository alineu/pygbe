#### Note: The following has been tested successfully on NU Discovery cluster

### 1. Connect to the Discovery cluster

	> ssh -X username@login-00.discovery.neu.edu

### 2. Install/load Anaconda

	> module load anaconda3/3.7
	> echo ". /shared/centos7/anaconda3/3.7/etc/profile.d/conda.sh" >> ~/.bashrc
	> source ~/.bashrc

### 3. Reserve a gpu/multigpu compute node

	> srun -p gpu --nodes=1 --pty --gres=gpu:1 --time=04:00:00 --export=ALL /bin/bash

### 4. Activate the environment

	> conda activate

If everything goes well, `(base)` will show up next to your `username@computenode` in shell, e.g.

    > (base)[username@c1234]

### 5. Load the required modules

	> module load boost/1.63.0
	> module load cuda/10.2

### 6. Create a conda environemnt for PyGBe (may take a while)

#### If you have attempted to install PyGBe using conda previously do this:

You need to delete the old environment to avoid possible version collisions. sure , First, delete the previously created `pygbe-env` environment from the old installation (You might have named it differently!). If that's the case, you can see your conda environments using

	> ls ~/.conda/envs

To remove the old environment you can

	> conda remove -n pygbe-env --all

or delete the env folder

	> rm -rf ~/.conda/envs/pygbe-env
	> conda create -n pygbe-env python=3.7 -y

#### If this is the first time you are installing PyGBe:

	> conda create -n pygbe-env python=3.7 -y

### 7. Activate the PyGBe environment

	> conda activate pygbe-env

### 8. Clone the PyGBe repository

If you have a preffered directory to maintain your repositories go to that directory and then

	> git clone https://github.com/alineu/pygbe3.git
	> cd pygbe

or

	> git clone git@github.com:alineu/pygbe3.git
	> cd pygbe

### 9. checkout the asymmetric_Py3 branch

	> git checkout asymmetric_Py3

**Note**: This is a critical step so make sure you don't miss it!

### 10. Install the remaining dependencies

#### 10.1 `numpy` and `scipy`

conda install -y numpy scipy

#### 10.2.`SWIG`

install `SWIG` using

	> mkdir src
	> cd src
	> wget https://versaweb.dl.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz
	> tar -xvzf swig-3.0.12.tar.gz
	> cd swig-3.0.12
	> ./configure --prefix=$PWD
	> make
	> make install

before installing pycuda, make sure there is absolutely no trace of previous installations left on your machine. So we delete them from the potential locations.

	> rm -rf ~/.conda/pkgs/pycuda*
	> rm -rf ~/.local/lib/python*/site-packages/pycuda*

Then install `pycuda` using

	> pip install -Iv pycuda

### 11. Install PyGBe and test the installation

#### 11.1 Install

	> cd ../../bem_pycuda/
	> make all

#### 11.2 Test

	> python main_asymmetric.py input_files/his.param input_files/his_stern.config --asymmetric --chargeForm

If the library is installed properly, you should see something like

    > Run started on:
    > 	Date: year/m/d
    > 	Time: hr:min:sec
    > 
    > Reading pqr for region 2 from ../geometry/his/his_prot.pqr
    > 
    > Reading surface 0 from file ../geometry/his/his_d01
    > Time load mesh: 0.036644
    > Removed areas=0: 0
    > 
    > Reading surface 1 from file ../geometry/his/his_d01_stern
    > Time load mesh: 0.041806
    > Removed areas=0: 81
    > 
    > Total elements : 1555
    > Total equations: 3110
    > 
    > ...
    > ...
    > ...
    > 
    > Totals:
    > Esolv = -19.045255 kcal/mol
    > Esurf = 0.000000 kcal/mol
    > Ecoul = -119.087367 kcal/mol

Time = 15.916039 s
### 12. Deactivate the environments whenever you're done:

	> conda deactivate
	> conda deactivate

twice!

### Note

Next time, when you connect to a GPU compute node you only need to run the following to enable and use `PyGBe` library:

#### Activate the Anaconda Module

	> module load anaconda3/3.7
	> conda activate

#### Activate the PyGBe Environment

	> conda activate pygbe-env

#### Load the Rest of the Modules

	> module load boost/1.63.0
	> module load cuda/10.2

#### Important
You might have to add the location of `pycuda` library to your `PATH` if `Python` fails to import the library. You can do this by adding

    > export PATH=$PATH:path/to/pycuda

to your `~/.bashrc` file (and then source it!).

`path_to_pycuda` is the location where the `pycuda` library is installed, e.g. `$HOME/.conda/envs/pygbe-env/lib/python3.7/site-packages/pycuda`

If you don't know the location of `pycuda` library you can find it using

	> find ~ -type d -name 'pycuda*'