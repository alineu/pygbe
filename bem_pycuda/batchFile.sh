#!/bin/bash
dirName=$1
if [ -d "../geometry/$dirName$" ]; then

    if [ -e "batch_$dirName.script" ]; then
        echo "File exists"
    else 
        echo "#SBATCH --nodes=1" >> batch_$dirName.script
        echo "#SBATCH --error=err_%j.txt" >> batch_$dirName.script
        echo "#SBATCH --output=out_%j.txt" >> batch_$dirName.script
        echo "#SBATCH --job-name=membrane" >> batch_$dirName.script
        echo "#SBATCH --partition=gpu" >> batch_$dirName.script
        echo "#SBATCH --time=24:00:00" >> batch_$dirName.script
        echo "#SBATCH --gres=gpu:1" >> batch_$dirName.script
        echo "#SBATCH --mail-type=ALL" >> batch_$dirName.script
        echo "#SBATCH --mail-user=mehdizadehrahimi.a@northeastern.edu" >> batch_$dirName.script
        echo "#SBATCH --mem=125Gb" >> batch_$dirName.script
        echo "python main_asymmetric_test.py input_files/membrane_Chris_Big_mem.param input_files/membrane_$dirName.config --asymmetric --chargeForm > ../geometry/$dirNam$" >> batch_$dirName.script
    fi
else
    echo "../geometry/$dirName$ does not exist!"
fi
