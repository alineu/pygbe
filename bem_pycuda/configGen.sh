#!/bin/bash
if [ -e "input_files/membrane_$1.config" ]; then
    echo "File exists"
else 
    echo "FILE    ../geometry/$1/memDiel     dielectric_layer" >> input_files/membrane_$1.config
    echo "FILE    ../geometry/$1/memStern    stern_interface" >> input_files/membrane_$1.config
    echo "--------------------------------" >> input_files/membrane_$1.config
    echo "PARAM   LorY E?     Dielec  kappa   charges?    coulomb?    charge_file                             Nparent     parent      Nchild	children" >> input_files/membrane_$1.config
    echo "FIELD   1    0      78.36      1e-12   0           0           NA                                      0           NA          1           1" >> input_files/membrane_$1.config
    echo "FIELD   1    0      78.36      1e-12   0           0           NA                                      1           1           1           0" >> input_files/membrane_$1.config
    echo "FIELD   1    1      1       1e-12   1           1           ../geometry/$1/membrane.pqr               1           0           0           NA" >> input_files/membrane_$1.config
fi
