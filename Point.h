#pragma once
template <class T>
class Point
{
protected:
	T x, y;

public:
	Point() : x(0.0), y(0.0){}
	Point(T _x) : x(_x), y(_x) {}
	Point(T _x, T _y) : x(_x), y(_y) {}
	Point(const Point& p)
	{
		this->x = p.x;
		this->y = p.y;
	}
	~Point() {}

public:
	void SetX(const T x)
	{
		this->x = x;
	}
	T GetX() const
	{
		return this->x;
	}
	void SetY(const T y)
	{
		this->y = y;
	}
	T GetY() const
	{
		return this->y;
	}
	void SetValue(const T x, const T y)
	{
		this->x = x;
		this->y = y;
	}
	void GetValue(T& x, T& y) const
	{
		x = this->x;
		y = this->y;
	}
};

