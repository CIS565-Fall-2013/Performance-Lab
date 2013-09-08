Performance Lab
=====================

Exercises to run in CIS 565 Performance Lab (09/25)

===================================================
USAGE
===================================================
WINDOWS
---------
Use included Visual Studio 2010 or 2012 project. Should run out of the box once CUDA is installed. The 2010 Project is configured for CUDA 4.0 and 2012 Project is configured for CUDA 5.0

To change the CUDA version, right click the project and click "Build Customizations". Choose the CUDA version you would like to work with.

LINUX
---------
Use Makefile to compile project. This can be changed to Mac very easily.
However, for the use of the Performance Lab, it would be preferable to use Windows since we will be going through the use of various tools provided for CUDA performance and debugging.
Requires gcc/g++ 4.4 (or nvcc compatible version)
CUDA 4.0 or later required (5.0 or later version recommended)

SOURCE FILES
---------
All Visual Studio projects and makefiles use the same source files located in the /src folder. So irrespective of which project you work with, the same files are edited and can be easily ported to any other version.

===================================================
EXERCISES
===================================================
CUDA BENCHMARK
---------------
This executable runs a benchmark for pageable and pinned Host->Device and Device->Host memory transfer using cudaMemcpy. It also runs a benchmark for global memory Device->Device memory copies.

It is a good way to assess the maximum practical bandwidth limits of GPUs.

TRANSPOSE
----------
Executes matrix transpose operations.
Each stage advances the performance of the kernel.

REDUCTION
----------
Executes sum of all elements in an array.
Each stage advances the performance of the kernel.

===================================================
NOTES
===================================================
* Reduction and Transpose exercises run CUDA BENCHMARK internally and display output.
* The VS projects and Makefiles are built for 64-bit machines. You can modify it for 32-bit fairly easily.
* CUDA 4.0 does not accept compute_30 code generation (Project Properties -> CUDA C/C++ -> Device). Use compute_20 instead.

===================================================
CONTACT
===================================================
Email: 

* pjcozzi+cis565@gmail.com
* liamboone+cis565@gmail.com
* shehzan@accelereyes.com

You may also post issues to the Github issue page.

===================================================
