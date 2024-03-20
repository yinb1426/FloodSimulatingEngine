#pragma once
#include "Utils.h"
#include <cuda_runtime.h>

__global__ void InitFlowFields(FlowField* flowField, FlowField* newFlowField, size_t sizeX, size_t sizeY);
__global__ void InitVelocity(Vec2* waterVelocity, size_t sizeX, size_t sizeY);
__global__ void UpdateSurfaceHeight(double* terrainHeight, double* buildingHeight, double* surfaceHeight, size_t sizeX, size_t sizeY);
__global__ void WaterIncrementByRainfall(double* waterHeight, double* rainfallRate, size_t sizeX, size_t sizeY, double deltaT, size_t numRainfallLayer, size_t step, size_t interval);
__global__ void WaterIncrementByRiverInflow(double* waterHeight, Vec2* waterVelocity, Vec3* riverInflow, size_t sizeX, size_t sizeY);
__global__ void UpdateOutputFlowField(FlowField* flowField, FlowField* newFlowField, double* surfaceHeight, double* waterHeight, size_t sizeX, size_t sizeY, double deltaT, double pipeLength, double gravity);
__global__ void UpdateNewFlowField(FlowField* flowField, FlowField* newFlowField, size_t sizeX, size_t sizeY);
__global__ void UpdateWaterVelocityAndHeight(double* waterHeight, Vec2* waterVelocity, FlowField* flowField, size_t sizeX, size_t sizeY, double deltaT, double pipeLength);
__global__ void Evaporation(double* waterHeight, size_t sizeX, size_t sizeY, double Ke, double deltaT);
__global__ void WaterHeightChangeByDrain(double* waterHeight, double* drainRate, size_t sizeX, size_t sizeY, double deltaT);

__global__ void SetOne(double* A, const size_t sizeX, const size_t sizeY);