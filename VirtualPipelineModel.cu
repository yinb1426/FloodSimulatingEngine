#include "VirtualPipelineModel.cuh"

__global__ void InitFlowFields(FlowField* flowField, FlowField* newFlowField, unsigned int sizeX, unsigned int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;
	flowField[idx].left   = 0.0;
	flowField[idx].right  = 0.0;
	flowField[idx].top    = 0.0;
	flowField[idx].bottom = 0.0;
	newFlowField[idx].left   = 0.0;
	newFlowField[idx].right  = 0.0;
	newFlowField[idx].top    = 0.0;
	newFlowField[idx].bottom = 0.0;
}

//__global__ void InitFlowFields(double* flowField, double* newFlowField, unsigned int sizeX, unsigned int sizeY)
//{
//	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int idx = iy * sizeX + ix;
//	if (idx >= sizeX * sizeY)
//		return;
//	flowField[0 + idx * 4] = idx * 1.0;
//	flowField[1 + idx * 4] = idx * 1.0;
//	flowField[2 + idx * 4] = idx * 1.0;
//	flowField[3 + idx * 4] = idx * 1.0;
//	newFlowField[0 + idx * 4] = idx * 1.0;
//	newFlowField[1 + idx * 4] = idx * 1.0;
//	newFlowField[2 + idx * 4] = idx * 1.0;
//	newFlowField[3 + idx * 4] = idx * 1.0;
//}

__global__ void InitVelocity(Vec2* waterVelocity, unsigned int sizeX, unsigned int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;
	waterVelocity[idx].x = 0.0;
	waterVelocity[idx].y = 0.0;
}

__global__ void UpdateSurfaceHeight(double* terrainHeight, double* buildingHeight, double* surfaceHeight, unsigned int sizeX, unsigned int sizeY) 
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;
	surfaceHeight[idx] = terrainHeight[idx] + buildingHeight[idx];
}

__global__ void WaterIncrementByRainfall(double* waterHeight, double* rainfallRate, unsigned int sizeX, unsigned int sizeY,
	double deltaT, unsigned int numRainfallLayer, unsigned int step, unsigned int interval)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;

	double currentRainfallRate = 0.0;
	double oldWaterHeight = waterHeight[idx];
	double ratio = 0.0;
	unsigned int currentLayerNum = int(step / interval);

	if (currentLayerNum >= numRainfallLayer - 1)
		currentRainfallRate = rainfallRate[(numRainfallLayer - 1) * sizeX * sizeY + idx];
	else
	{
		ratio = (step % interval) / interval * 1.0;
		currentRainfallRate = rainfallRate[currentLayerNum * sizeX * sizeY + idx] * (1.0 - ratio) + rainfallRate[(currentLayerNum + 1) * sizeX * sizeY + idx] * ratio;
	}

	waterHeight[idx] = oldWaterHeight + deltaT * currentRainfallRate / 600.0;
}

__global__ void WaterIncrementByRiverInflow(double* waterHeight, Vec2* waterVelocity, Vec3* riverInflow, unsigned int sizeX, unsigned int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;
	if (riverInflow[idx].x >= 0.0)
	{
		waterHeight[idx] = riverInflow[idx].x;
		waterVelocity[idx].x = riverInflow[idx].y;
		waterVelocity[idx].y = riverInflow[idx].z;
	}
}

__global__ void UpdateOutputFlowField(FlowField* flowField, FlowField* newFlowField, double* surfaceHeight, double* waterHeight, unsigned int sizeX, unsigned int sizeY,
	double deltaT, double pipeLength, double gravity)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (!(ix >= 1 && ix < sizeX - 1 && iy >= 1 && iy < sizeY - 1))
		return;
	double deltaHeightLeft   = surfaceHeight[idx] + waterHeight[idx] - surfaceHeight[iy * sizeX + (ix - 1)] - waterHeight[iy * sizeX + (ix - 1)];
	double deltaHeightRight  = surfaceHeight[idx] + waterHeight[idx] - surfaceHeight[iy * sizeX + (ix + 1)] - waterHeight[iy * sizeX + (ix + 1)];
	double deltaHeightTop    = surfaceHeight[idx] + waterHeight[idx] - surfaceHeight[(iy - 1) * sizeX + ix] - waterHeight[(iy - 1) * sizeX + ix];
	double deltaHeightBottom = surfaceHeight[idx] + waterHeight[idx] - surfaceHeight[(iy + 1) * sizeX + ix] - waterHeight[(iy + 1) * sizeX + ix];
	
	double oldOutputFlowLeft   = flowField[idx].left;
	double oldOutputFlowRight  = flowField[idx].right;
	double oldOutputFlowTop    = flowField[idx].top;
	double oldOutputFlowBottom = flowField[idx].bottom;

	double damping = 0.9999;

	double newOutputFlowLeft   = fmax(0.0, damping * oldOutputFlowLeft   + deltaT * pipeLength * pipeLength * gravity * deltaHeightLeft   / pipeLength);
	double newOutputFlowRight  = fmax(0.0, damping * oldOutputFlowRight  + deltaT * pipeLength * pipeLength * gravity * deltaHeightRight  / pipeLength);
	double newOutputFlowTop    = fmax(0.0, damping * oldOutputFlowTop    + deltaT * pipeLength * pipeLength * gravity * deltaHeightTop    / pipeLength);
	double newOutputFlowBottom = fmax(0.0, damping * oldOutputFlowBottom + deltaT * pipeLength * pipeLength * gravity * deltaHeightBottom / pipeLength);

	double outputVolume = (newOutputFlowLeft + newOutputFlowRight + newOutputFlowTop + newOutputFlowBottom) * deltaT;

	double K = fmin(1.0, waterHeight[idx] * pipeLength * pipeLength / outputVolume);

	newOutputFlowLeft   *= K;
	newOutputFlowRight  *= K;
	newOutputFlowTop    *= K;
	newOutputFlowBottom *= K;

	newFlowField[idx].left   = newOutputFlowLeft;
	newFlowField[idx].right  = newOutputFlowRight;
	newFlowField[idx].top    = newOutputFlowTop;
	newFlowField[idx].bottom = newOutputFlowBottom;
}

__global__ void UpdateNewFlowField(FlowField* flowField, FlowField* newFlowField, unsigned int sizeX, unsigned int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;

	flowField[idx].left   = newFlowField[idx].left;
	flowField[idx].right  = newFlowField[idx].right;
	flowField[idx].top    = newFlowField[idx].top;
	flowField[idx].bottom = newFlowField[idx].bottom;

	//newFlowField[idx].left   = 0.0;
	//newFlowField[idx].right  = 0.0;
	//newFlowField[idx].top    = 0.0;
	//newFlowField[idx].bottom = 0.0;
}

__global__ void UpdateWaterVelocityAndHeight(double* waterHeight, Vec2* waterVelocity, FlowField* flowField,  unsigned int sizeX, unsigned int sizeY,
	double deltaT, double pipeLength)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (!(ix >= 1 && ix < sizeX - 1 && iy >= 1 && iy < sizeY - 1))
		return;
	
	double oldWaterHeight = waterHeight[idx];
	double deltaV = (flowField[iy * sizeX + (ix - 1)].right + flowField[iy * sizeX + (ix + 1)].left + flowField[(iy - 1) * sizeX + ix].bottom + flowField[(iy + 1) * sizeX + ix].top - flowField[idx].left - flowField[idx].right - flowField[idx].top - flowField[idx].bottom) * deltaT;
	double d2 = oldWaterHeight + deltaV / (pipeLength * pipeLength);
	double avgWaterHeight = (d2 + oldWaterHeight) / 2.0;
	double velocityFactor = avgWaterHeight * pipeLength;

	double deltaWX = (flowField[iy * sizeX + (ix - 1)].right + flowField[idx].right - flowField[idx].left - flowField[iy * sizeX + (ix + 1)].left) / 2.0;
	double deltaWY = (flowField[(iy - 1) * sizeX + ix].bottom + flowField[idx].bottom - flowField[idx].top - flowField[(iy + 1) * sizeX + ix].top) / 2.0;

	double velocityU = 0.0; 
	double velocityV = 0.0;

	if (velocityFactor > 5e-7)
	{
		velocityU = deltaWX / velocityFactor;
		velocityV = deltaWY / velocityFactor;
	}
	waterHeight[idx] = d2;
	waterVelocity[idx].x = velocityU;
	waterVelocity[idx].y = velocityV;
}

__global__ void Evaporation(double* waterHeight, unsigned int sizeX, unsigned int sizeY, double Ke, double deltaT)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;

	double oldWaterHeight = waterHeight[idx];
	double newWaterHeight = oldWaterHeight * (1.0 - Ke * deltaT);
	if (newWaterHeight < 0.00005)
		waterHeight[idx] = 0.0;
	else
		waterHeight[idx] = newWaterHeight;
}
__global__ void WaterHeightChangeByDrain(double* waterHeight, double* drainRate, unsigned int sizeX, unsigned int sizeY, double deltaT)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;

	double oldWaterHeight = waterHeight[idx];
	double newWaterHeight = oldWaterHeight + deltaT * drainRate[idx];
	if (newWaterHeight < 0.00005)
		waterHeight[idx] = 0.0;
	else
		waterHeight[idx] = newWaterHeight;
}

__global__ void SetOne(double* A, unsigned int sizeX, unsigned int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;
	if (idx >= sizeX * sizeY)
		return;
	A[idx] = 1.0;
}