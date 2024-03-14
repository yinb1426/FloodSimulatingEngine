#pragma once

#include <vector>
#include <string>
#include <fstream>
typedef struct FlowField
{
	double left = 0.0, right = 0.0, top = 0.0, bottom = 0.0;
}FlowField;

typedef struct Vector2
{
	double x = 0.0, y = 0.0;
}Vec2;

typedef struct Vector3
{
	double x = 0.0, y = 0.0, z = 0.0;
}Vec3;