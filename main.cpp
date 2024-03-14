//#include <iostream>
//#include <vector>
//#include <cuda_runtime.h>
//#include "Utils.h"
//#include "VirtualPipelineModel.cuh"
//
//using namespace std;
//
//int main()
//{
//	dim3 dimBlock(4, 3);
//	dim3 dimGrid(1, 1);
//
//	unsigned int sizeX = 4, sizeY = 3;
//
//	vector<double> A(12);
//	vector<double> B(12);
//	double* gA;
//
//	cudaMalloc((void**)&gA, sizeof(double) * 12);
//	cudaMemcpy(gA, &A[0], sizeof(double) * 12, cudaMemcpyHostToDevice);
//
//	SetOne << < dimGrid, dimBlock >> > (gA, sizeX, sizeY);
//
//	cudaMemcpy(&A[0], gA, sizeof(double) * 12, cudaMemcpyDeviceToHost);
//
//	for (int j = 0; j < sizeY; ++j)
//	{
//		for (int i = 0; i < sizeX; ++i)
//			cout << A[j * sizeX + i] << ",";
//		cout << endl;
//	}
//	cudaFree(gA);
//	return 0;
//}