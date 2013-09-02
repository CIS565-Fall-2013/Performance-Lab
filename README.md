Performance Lab
=====================

Exercises to run in CIS 565 Performance Lab (09/25)

===================================================
USAGE
===================================================
LINUX
---------
Use Makefile to compile project.
Requires gcc/g++ 4.4 (or nvcc compatible version)
CUDA 4.0 or later required (5.0 or later version recommended)

WINDOWS
---------
Use included Visual Studio 2010 or 2012 project. Should run out of the box once CUDA is installed.

===================================================
EXERCISES
===================================================
CUDA BENCHMARK
---------------
This executable runs a benchmark for pageable and pinned Host->Device and Device->Host memory transfer using cudaMemcpy. It also runs a benchmark for global memory Device->Device memory copies.

It is a good way to assess the maximum practical bandwidth limits of GPUS.

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
* Reduction and Transpose exercises run CUDA BENCHMARK internally and display output
* The VS projects and Makefiles are built for 64-bit machines. You can modify it for 32-bit fairly easily.

===================================================
CONTACT
===================================================
Email: 

* pjcozzi+cis565@gmail.com
* liamboone+cis565@gmail.com
* shehzan@accelereyes.com

You may also post issues to the Github issue page.

===================================================
