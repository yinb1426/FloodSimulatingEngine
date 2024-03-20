#include "Model.h"

void Model::SetSizeX(const size_t sizeX)
{
    this->sizeX = sizeX;
}

size_t Model::GetSizeX() const
{
    return this->sizeX;
}

void Model::SetSizeY(const size_t sizeY)
{
    this->sizeY = sizeY;
}

size_t Model::GetSizeY() const
{
    return this->sizeY;
}

void Model::SetDeltaT(const double deltaT)
{
    this->deltaT = deltaT;
}

double Model::GetDeltaT() const
{
    return this->deltaT;
}

void Model::SetTerrainHeight(const vector<double> terrainHeight)
{
    this->terrainHeight = terrainHeight;
}

vector<double> Model::GetTerrainHeight() const
{
    return this->terrainHeight;
}

void Model::SetWaterHeight(const vector<double> waterHeight)
{
    this->waterHeight = waterHeight;
}

vector<double> Model::GetWaterHeight() const
{
    return this->waterHeight;
}