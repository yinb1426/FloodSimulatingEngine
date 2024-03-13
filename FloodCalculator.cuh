#ifndef FLOOD_CALCULATOR_CUH
#define FLOOD_CALCULATOR_CUH
#include "Utils.h"
#include <cuda_runtime.h>

__global__ void InitFlowFields(FlowField* flowField, FlowField* newFlowField, unsigned int sizeX, unsigned int sizeY);
__global__ void InitVelocity(Vec2* waterVelocity, unsigned int sizeX, unsigned int sizeY);
__global__ void WaterIncrementByRainfall(double* waterHeight, double* rainfallRate, unsigned int sizeX, unsigned int sizeY, double deltaT, unsigned int numRainfallLayer, unsigned int step, unsigned int interval);
__global__ void WaterIncrementByRiverInflow(double* waterHeight, Vec2* waterVelocity, Vec3* riverInflow, unsigned int sizeX, unsigned int sizeY);
__global__ void UpdateOutputFlowField(FlowField* flowField, FlowField* newFlowField, double* surfaceHeight, double* waterHeight, unsigned int sizeX, unsigned int sizeY, double deltaT, double pipeLength, double gravity);
__global__ void UpdateNewFlowField(FlowField* flowField, FlowField* newFlowField, unsigned int sizeX, unsigned int sizeY);
__global__ void UpdateWaterVelocityAndHeight(double* waterHeight, Vec2* waterVelocity, FlowField* flowField, unsigned int sizeX, unsigned int sizeY, double deltaT, double pipeLength);
__global__ void Evaporation(double* waterHeight, unsigned int sizeX, unsigned int sizeY, double Ke, double deltaT);
__global__ void WaterHeightChangeByDrain(double* waterHeight, double* drainRate, unsigned int sizeX, unsigned int sizeY, double deltaT);


#endif