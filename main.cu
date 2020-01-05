#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
	{
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float* A, float* B, float* C, int n)
{
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void**)&d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_C, size);

	vecAddKernel <<< ceil(n / 256.0), 256 >>> (d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main()
{
	float A[5] = { 1, 2, 3, 4, 5 };
	float B[5] = { 10, 20, 30, 40, 50 };
	float C[5];

	cout << "A:";
	for (int i = 0; i < 5; i++)
	{
		cout << A[i] << " ";
	}
	cout << endl;

	cout << "B:";
	for (int i = 0; i < 5; i++)
	{
		cout << B[i] << " ";
	}
	cout << endl;

	vecAdd(A, B, C, 5);

	cout << "C:";
	for (int i = 0; i < 5; i++)
	{
		cout << C[i] << " ";
	}
	cout << endl;


	// cout << "Hello" << endl;
}