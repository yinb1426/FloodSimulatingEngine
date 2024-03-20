#include "VirtualPipelineModel.cuh"
#include "VPMCalculator.cuh"

void VPM::SetPipeLength(const double newPipeLength)
{
	this->pipeLength = newPipeLength;
}

double VPM::GetPipeLength() const
{
	return this->pipeLength;
}

void VPM::SetGravity(const double newGravity)
{
	this->gravity = newGravity;
}

double VPM::GetGravity() const
{
	return this->gravity;
}

void VPM::SetKe(const double newKe)
{
	this->Ke = newKe;
}

double VPM::GetKe() const
{
	return this->Ke;
}

void VPM::InitDevice()
{
	cudaMalloc((void**)&gTerrainHeight, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gBuildingHeight, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gDamHeight, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gSurfaceHeight, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gWaterHeight, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gRainfallRate, sizeof(double) * numRainfallLayer * sizeX * sizeY);
	cudaMalloc((void**)&gDrainRate, sizeof(double) * sizeX * sizeY);
	cudaMalloc((void**)&gRiverInflow, sizeof(Vec3) * sizeX * sizeY);
	cudaMalloc((void**)&gWaterVelocity, sizeof(Vec2) * sizeX * sizeY);
	cudaMalloc((void**)&gFlowField, sizeof(FlowField) * sizeX * sizeY);
	cudaMalloc((void**)&gNewFlowField, sizeof(FlowField) * sizeX * sizeY);
}

void VPM::SendAllDataToDevice()
{
	cudaMemcpy(gTerrainHeight, &terrainHeight[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gBuildingHeight, &buildingHeight[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gDamHeight, &damHeight[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gSurfaceHeight, &surfaceHeight[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gWaterHeight, &waterHeight[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gRainfallRate, &rainfallRate[0], sizeof(double) * numRainfallLayer * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gDrainRate, &drainRate[0], sizeof(double) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gRiverInflow, &riverInflow[0], sizeof(Vec3) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gWaterVelocity, &waterVelocity[0], sizeof(Vec2) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gFlowField, &flowField[0], sizeof(FlowField) * sizeX * sizeY, cudaMemcpyHostToDevice);
	cudaMemcpy(gNewFlowField, &newFlowField[0], sizeof(FlowField) * sizeX * sizeY, cudaMemcpyHostToDevice);
}

void VPM::PreparationForSimulaion()
{
	dim3 dimBlock(32, 16);
	dim3 dimGrid((sizeX - 1) / dimBlock.x + 1, (sizeY - 1) / dimBlock.y + 1);

	InitFlowFields << < dimGrid, dimBlock >> > (gFlowField, gNewFlowField, sizeX, sizeY);
	InitVelocity << < dimGrid, dimBlock >> > (gWaterVelocity, sizeX, sizeY);
	UpdateSurfaceHeight << < dimGrid, dimBlock >> > (gTerrainHeight, gBuildingHeight, gSurfaceHeight, sizeX, sizeY);
}

void VPM::RunSimulation(const size_t step)
{
	dim3 dimBlock(32, 16);
	dim3 dimGrid((sizeX - 1) / dimBlock.x + 1, (sizeY - 1) / dimBlock.y + 1);

	WaterIncrementByRainfall << < dimGrid, dimBlock >> > (gWaterHeight, gRainfallRate, sizeX, sizeY, deltaT, numRainfallLayer, step, 3000);
	UpdateOutputFlowField << < dimGrid, dimBlock >> > (gFlowField, gNewFlowField, gSurfaceHeight, gWaterHeight, sizeX, sizeY, deltaT, pipeLength, gravity);
	UpdateNewFlowField << < dimGrid, dimBlock >> > (gFlowField, gNewFlowField, sizeX, sizeY);
	UpdateWaterVelocityAndHeight << < dimGrid, dimBlock >> > (gWaterHeight, gWaterVelocity, gFlowField, sizeX, sizeY, deltaT, pipeLength);
	//Evaporation << < dimGrid, dimBlock >> > (gWaterHeight, sizeX, sizeY, Ke, deltaT);
}

void VPM::GetResultFromDevice()
{
	cudaMemcpy(&waterHeight[0], gWaterHeight, sizeof(double) * sizeX * sizeY, cudaMemcpyDeviceToHost);
	cudaMemcpy(&waterVelocity[0], gWaterVelocity, sizeof(Vec2) * sizeX * sizeY, cudaMemcpyDeviceToHost);
}

void VPM::FreeAllData()
{
	cudaFree(gTerrainHeight);
	cudaFree(gBuildingHeight);
	cudaFree(gDamHeight);
	cudaFree(gSurfaceHeight);
	cudaFree(gWaterHeight);
	cudaFree(gRainfallRate);
	cudaFree(gDrainRate);
	cudaFree(gRiverInflow);
	cudaFree(gWaterVelocity);
	cudaFree(gFlowField);
	cudaFree(gNewFlowField);
}
