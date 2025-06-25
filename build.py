import os
from shutil import copy
from glob import glob
from pathlib import Path
import shutil


def compile_library():

    #Compile the library
    main_dir = os.getcwd()
    os.chdir(main_dir + '/code/cpp')
    #Delete the build folder if it exists:
    if os.path.exists('build'):
        shutil.rmtree('build')
    # Create the build folder:
    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    os.system("cmake ..")
    os.system("cmake --build . --config Release")

    #Copy the library to the build folder:
    for file in glob('Release/*.lib'):
        copy(file, main_dir + '/code/cpp')


if __name__ == '__main__':
    """Building script for Windows"""
    compile_library()