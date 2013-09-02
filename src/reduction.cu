#include <stdio.h>
#include <iostream>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//NVTX Dir: C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt
#include <nvToolsExt.h>

const int n_elements = 32 * 1024 * 1024;  // number of elements to reduce

using namespace std;

struct DIMS
{
	dim3 dimBlock;
	dim3 dimGrid;
};

struct DIMS2
{
	int dimThreads;
	int dimBlocks;
};

#define CUDA(call) do {                                 \
	cudaError_t e = (call);								\
	if (e == cudaSuccess) break;						\
	fprintf(stderr, __FILE__":%d: %s (%d)\n",			\
			__LINE__, cudaGetErrorString(e), e);		\
	exit(1);											\
} while (0)

inline unsigned divup(unsigned n, unsigned div)
{
	return (n + div - 1) / div;
}

double diffclock( clock_t clock1, clock_t clock2 )
{
	double diffticks = clock1 - clock2;
	double diffms    = diffticks / ( CLOCKS_PER_SEC / 1000.0);
	return diffms;
}

// Check errors
void postprocess(const int *ref, const int *res, int n)
{
	bool passed = true;
	for (int i = 0; i < n; i++)
	{
		if (res[i] != ref[i])
		{
		  printf("ID:%d \t Res:%d \t Ref:%d\n", i, res[i], ref[i]);
		  printf("%25s\n", "*** FAILED ***");
		  passed = false;
		  break;
		}
	}
	if(passed)
		printf("Post process PASSED!!!\n");
}

void preprocess(float *res, float *dev_res, int n)
{
	for (int i = 0; i < n; i++)
	{
		res[i] = -1;
	}
	cudaMemset(dev_res, -1, n * sizeof(float));
}

__global__ void copyKernel(const int* __restrict__ const a,
						   int* __restrict__ const b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;  // index
	b[i] = a[i];									// copy
}

static int reduce_cpu(int *data, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += data[i];
	return sum;
}

// INTERLEAVED ADDRESSING
// TODO put your kernel here
__global__ void reduce_stage0(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Copy input data to shared memory

	//Reduce within block

	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

// INTERLEAVED ADDRESSING NON DIVERGENT
// TODO put your kernel here
__global__ void reduce_stage1(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = 0;

	//Copy input data to shared memory

	//Reduce within block with coalesced indexing pattern

	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

// WARP MANAGEMENT WITHOUT BANK CONFLICT
// TODO put your kernel here
__global__ void reduce_stage2(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = 0;

	//Copy input data to shared memory

	//Reduce within block with coalesced indexing pattern and avoid bank conflicts

	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

// ADD DURING LOAD - USE HALF THE BLOCKS
// TODO put your kernel here
const int stage3_TILE = 2;
__global__ void reduce_stage3(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = 0;						//EACH BLOCK DOES WORK OF stage3_TILE*blockDim.x NO. OF ELEMENTS

	//Copy input data to shared memory. Add on load.

	//Reduce within block with coalesced indexing pattern and avoid bank conflicts
	//HINT: This part is same as stage2

	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

// WARP LOOP UNROLLING
// TODO put your kernel here
__device__ void stage4_warpReduce(volatile int* smem, int tid)
{
	//Write code for warp reduce here
}

const int stage4_TILE = 2;	//Tune this
__global__ void reduce_stage4(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = 0;						//EACH BLOCK DOES WORK OF stage4_TILE*blockDim.x NO. OF ELEMENTS

	//Copy input data to shared memory. Add on load.
	
	//Reduce within block with coalesced indexing pattern and avoid bank conflicts
	//HINT: This part is similar to stage3, is different in terms of warp reduction
	//Use stage4_warpReduce

	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

// COMPLETELY UNROLLED BLOCKS - USING TEMPLATES
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* smem, int tid)
{
	//Write code for warp reduce here. Same has stage 4 warp reduce
}

// TODO put your kernel here
const int stage5_TILE = 2;		//Tune this
template<unsigned int blockSize>
__global__ void reduce_stage5(int* d_idata, int* d_odata, int n)
{
	//Dynamic allocation of shared memory - See kernel call in host code
	extern __shared__ int smem[];

	//Calculate index
	int idx = 0;						//EACH BLOCK DOES WORK OF stage4_TILE*blockDim.x NO. OF ELEMENTS

	//Copy input data to shared memory. Add on load.
	
	//Reduce within block with coalesced indexing pattern and avoid bank conflicts
	//HINT: Explicity unroll the loop

	//Call the correct warpReduce
	
	//Copy result of reduction to global memory
	if(threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

int main()
{
	//Run Memcpy benchmarks
	nvtxRangeId_t cudaBenchmark = nvtxRangeStart("CUDA Memcpy Benchmark");
#if defined WIN64
	system(".\\..\\bin\\cudaBenchmark.exe");
#elif defined LINUX
	system("./bin/cudaBenchmark");
#endif
	nvtxRangeEnd(cudaBenchmark);
	///////////////////////////////////////////////////////////////////////////////////////////////

	//Allocate memory and initialize elements
	unsigned bytes = n_elements * sizeof(int);
	int *h_idata;
	CUDA( cudaMallocHost((void**)&h_idata, bytes) );		//Using Pinned Memory
	for (int i=0; i < n_elements; i++)
		h_idata[i] = (int)(rand() & 0xFF);

	// copy data to device memory
	int *d_idata = NULL;
	CUDA(cudaMalloc((void **) &d_idata, bytes));
	CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

	//Compute Gold Standard using CPU
	const int gold_result = reduce_cpu(h_idata, n_elements);

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

#define CPU_REDUCE
#ifdef CPU_REDUCE
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***CPU Reduce***" << endl;
	{
		// start the timer
		int cpu_result = -1;
		nvtxRangeId_t cpuBenchmark = nvtxRangeStart("CPU Reduce Benchmark");

		clock_t begin = clock();
		int iters = 100;
		for (int k=0; k<iters; k++)
		{
			cpu_result = reduce_cpu(h_idata, n_elements);
		}
		// stop the timer
		clock_t end = clock();
		nvtxRangeEnd(cpuBenchmark);

		float time = 0.0f;
		time = (float)diffclock(end, begin);

		// print out the time required for the kernel to finish the transpose operation
		//double Bandwidth = (double)iters*2.0*1000.0*(double)(bytes) / (1000.0*1000.0*1000.0*time);
		cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
		//cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

		//Check Result
		if(cpu_result == gold_result)
			cout << "Post process check PASSED!!!" << endl;
		else
			cout << "Post process check FAILED:-(" << endl;
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

	//Compute Pinned Memory Copy Benchmark
	cout << "******************************************" << endl;
	cout << "***Device To Device Copy***" << endl;
	{
		// Assign a 1D distribution of threads per blocks
		// Calculate number of blocks required

		DIMS2 dims;
		dims.dimThreads = 1024;
		dims.dimBlocks  = divup(n_elements, dims.dimThreads);

		// start the timer
		nvtxRangeId_t d2dBenchmark = nvtxRangeStart("Device to Device Copy");
		cudaEventRecord( start, 0);
		int *d_odata;
		CUDA( cudaMalloc((void **) &d_odata, bytes) );
		int iters = 100;
		for (int i=0; i<iters; i++)
		{
			// Launch the GPU kernel
			copyKernel<<<dims.dimBlocks, dims.dimThreads>>>(d_idata, d_odata);
		}
		// stop the timer
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		nvtxRangeEnd(d2dBenchmark);

		float time = 0.0f;
		cudaEventElapsedTime( &time, start, stop);

		// print out the time required for the kernel to finish the transpose operation
		double Bandwidth = (double)iters*2.0*1000.0*(double)(bytes) /
			(1000.0*1000.0*1000.0*time);	//3.0 for read of A and read and write of B
		cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
		cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

		// copy the answer back to the host (CPU) from the device (GPU)
		int *h_odata; CUDA( cudaMallocHost((void**)&h_odata, bytes) );
		cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

		postprocess(h_idata, h_odata, n_elements);
		CUDA( cudaFreeHost(h_odata) );
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////

#if 0
	//Compute GPU Reduce Benchmarks
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 0***" << endl;
	{
		//Calculate Threads per block and total blocks required
		//HINT: Look at copy kernel dims computed above
		DIMS2 dims;
		dims.dimThreads = 1;		//Start with any (preferable 2^n) threads per block. Then tune once working.
		dims.dimBlocks  = 1;
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;

		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements
		reduce_stage0<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];

		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				reduce_stage0<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
		
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

#if 0
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 1***" << endl;
	{
		//Calculate Threads per block and total blocks required
		DIMS2 dims;
		dims.dimThreads = 1;		//Start with any (preferable 2^n) threads per block. Then tune once working.
		dims.dimBlocks  = 1;
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;
		
		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements
		reduce_stage1<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];

		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				reduce_stage1<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
		
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

#if 0
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 2***" << endl;
	{
		//Calculate Threads per block and total blocks required
		DIMS2 dims;
		dims.dimThreads = 1;		//Start with any (preferable 2^n) threads per block. Then tune once working.
		dims.dimBlocks  = 1;
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;

		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements
		reduce_stage2<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];

		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				reduce_stage2<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
		
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

#if 0
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 3***" << endl;
	{
		//Calculate Threads per block and total blocks required
		DIMS2 dims;
		dims.dimThreads = 1;		//Start with any (preferable 2^n) threads per block. Then tune once working.
		dims.dimBlocks  = 1;		//Don't forget to take TILE into account while computing blocks
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;

		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements
		reduce_stage3<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];

		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				reduce_stage3<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
		
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

#if 0
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 4***" << endl;
	{
		//Calculate Threads per block and total blocks required
		DIMS2 dims;
		dims.dimThreads = 1;		//Start with any (preferable 2^n) threads per block. Then tune once working.
		dims.dimBlocks  = 1;		//Don't forget to take TILE into account while computing blocks
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;

		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements
		reduce_stage4<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];
		
		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				reduce_stage4<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

#if 0
	////////////////////////////////////////////////////////////
	cout << "******************************************" << endl;
	cout << "***Reduction Stage 5***" << endl;
	{
		//Calculate Threads per block and total blocks required
		DIMS2 dims;
		const int threads = 1;		//We can use this in templates
		dims.dimThreads = threads;
		dims.dimBlocks  = 1;		//Don't forget to take TILE into account while computing blocks
		printf("Elements %u   Blocks %u   Threads %u\n", n_elements, dims.dimBlocks, dims.dimThreads);

		//Do once for error checking
		int gpu_result = 0;

		CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
		size_t block_bytes = dims.dimBlocks * sizeof(int);
		int *d_odata = NULL;
		CUDA(cudaMalloc((void**)&d_odata, block_bytes));
		CUDA(cudaMemset(d_odata, 0, block_bytes));
		
		// TODO call your reduce kernel(s) with the right parameters
		// INPUT:       d_idata
		// OUTPUT:      d_odata
		// ELEMENTS:    n
		// (1) reduce across all elements

		//Don't forget to add the template
		//reduce_stage5<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);

		// (2) reduce across all blocks -> Choose between CPU/GPU
		int *h_blocks = (int *)malloc(block_bytes);
		CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
		for (int i = 0; i < dims.dimBlocks; ++i)
			gpu_result += h_blocks[i];

		printf("gpu %u   gold %u   \n", gpu_result, gold_result);
		printf("Post process: ");
		printf((gpu_result==gold_result) ? "PASSED!!!\n" : "FAILED:-(\n");

		if(gpu_result == gold_result)
		{
			//Start Benchmark
			int iters = 100;
			CUDA(cudaEventRecord(start, 0));
			for(int i = 0; i < iters; i++)
			{
				//Don't forget to add the template
				//reduce_stage5<<<dims.dimBlocks, dims.dimThreads/*,Declare appropriate shared memory space*/>>>(d_idata, d_odata, n_elements);
				CUDA( cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost) );
				for (int i = 0; i < dims.dimBlocks; ++i)
					gpu_result += h_blocks[i];
			}
			CUDA(cudaEventRecord(stop, 0));
			CUDA(cudaEventSynchronize(stop));
			
			float time_ms;
			// that's the time your kernel took to run in ms!
			CUDA(cudaEventElapsedTime(&time_ms, start, stop));
			double Bandwidth = iters * 1e-9 * bytes / (time_ms / 1e3);
			cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
			printf("bandwidth %.2f GB/s\n", Bandwidth);
		}
		free(h_blocks);
		cudaFree(d_odata);
	}
	cout << "******************************************" << endl;
	cout << endl;
	////////////////////////////////////////////////////////////
#endif

	////////////////////////////////////////////////////////////

	//CLEANUP
	CUDA( cudaEventDestroy(start) );
	CUDA( cudaEventDestroy(stop ) );
	
	CUDA( cudaFreeHost(h_idata) );
	CUDA( cudaFree(d_idata) );

	return 0;
}
