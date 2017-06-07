#pragma once

#include "Vec2.h"

namespace ettention
{
    template<typename _ValType>
    class Vec3
    {
    public:
		union {
			struct
			{
				_ValType x;
				_ValType y;
				_ValType z;
			};
			_ValType dim[3];
		};

		Vec3() {};
		Vec3(_ValType x, _ValType y, _ValType z) : x(x), y(y), z(z) {};
		Vec3(const Vec2<_ValType>& xy, _ValType z) : x(xy.x), y(xy.y), z(z) {};
		Vec3(_ValType x, const Vec2<_ValType>& yz) : x(x), y(yz.x), z(yz.y) {};
		~Vec3() {};

		double getLength() const
		{ 
			return std::sqrt((double)dotProduct(*this, *this)); 
		};

		float getLengthF() const 
		{ 
			return (float)this->getLength(); 
		};

		std::string to_string() const
		{
			return std::to_string(x) + "/" + std::to_string(y) + "/" + std::to_string(z);
		};

		Vec3 makeInverse() const 
		{
			double x = 1.0 / (double)this->x;
			double y = 1.0 / (double)this->y;
			double z = 1.0 / (double)this->z;
			return Vec3((_ValType)x, (_ValType)y, (_ValType)z);
		};

		Vec2<_ValType> xy() const { return Vec2<_ValType>(x, y); };
		Vec2<_ValType> yz() const { return Vec2<_ValType>(y, z); };
		Vec2<_ValType> xz() const { return Vec2<_ValType>(x, z); };

		const _ValType& operator[](unsigned int index) const { return dim[index]; };
        _ValType& operator[](unsigned int index) { return dim[index]; }; 

		Vec3& operator+=(const Vec3& other) 
		{
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		};

        Vec3& operator-=(const Vec3& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		};

        Vec3& operator*=(const Vec3& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;
			return *this;
		};

        Vec3& operator/=(const Vec3& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;
			return *this;
		};

        Vec3& operator*=(_ValType scalar)
		{
			x *= scalar;
			y *= scalar;
			z *= scalar;
			return *this;
		};

        Vec3& operator/=(_ValType scalar)
		{
			x /= scalar;
			y /= scalar;
			z /= scalar;
			return *this;
		};

		Vec3 operator+(const Vec3& other) const
		{
			return Vec3(x + other.x, y + other.y, z + other.z);
		};

        Vec3 operator-(const Vec3& other) const
		{
			return Vec3(x - other.x, y - other.y, z - other.z);
		};

        Vec3 operator*(const Vec3& other) const 
		{
			return Vec3(x * other.x, y * other.y, z * other.z);
		};

        Vec3 operator/(const Vec3& other) const
		{
			return Vec3(x / other.x, y / other.y, z / other.z);
		};

        Vec3 operator-() const
		{
			return Vec3(-x, -y, -z);
		};

        Vec3 operator*(_ValType scalar) const
		{
			return Vec3(x * scalar, y * scalar, z * scalar);
		};

        Vec3 operator/(_ValType scalar) const
		{
			return Vec3(x / scalar, y / scalar, z / scalar);
		};

        bool operator==(const Vec3& other) const
		{
			return x == other.x && y == other.y && z == other.z;
		};

        bool operator!=(const Vec3& other) const
		{
			return x != other.x || y != other.y || z != other.z;
		}

        template<typename _CastValue>
        operator Vec3<_CastValue>() const
        {
            return Vec3<_CastValue>((_CastValue)x, (_CastValue)y, (_CastValue)z);
        }

        // the following compare functions assume coordinate priorities z,y,x
        bool operator<(const Vec3& other) const
		{
			if (z < other.z)
				return true;
			if (z > other.z)
				return false;
			if (y < other.y)
				return true;
			if (y > other.y)
				return false;
			if (x < other.x)
				return true;
			return false;
		};

        bool operator<=(const Vec3& other) const
		{
			if (z < other.z)
				return true;
			if (z > other.z)
				return false;
			if (y < other.y)
				return true;
			if (y > other.y)
				return false;
			if (x <= other.x)
				return true;
			return false;
		};

        bool operator>(const Vec3& other) const
		{
			if (z > other.z)
				return true;
			if (z < other.z)
				return false;
			if (y > other.y)
				return true;
			if (y < other.y)
				return false;
			if (x > other.x)
				return true;
			return false;
		};

        bool operator>=(const Vec3& other) const
		{
			if (z > other.z)
				return true;
			if (z < other.z)
				return false;
			if (y > other.y)
				return true;
			if (y < other.y)
				return false;
			if (x >= other.x)
				return true;
			return false;
		};

		std::string string() const
		{
			std::stringstream sstream;
			sstream << *this;
			return sstream.str();
		};

    };

    template<typename _ValType>
    _ValType dotProduct(const Vec3<_ValType>& left, const Vec3<_ValType>& right)
    {
        return left.x * right.x + left.y * right.y + left.z * right.z;
    }

    template<typename _ValType>
    Vec3<_ValType> crossProduct(const Vec3<_ValType>& left, const Vec3<_ValType>& right)
    {
        return Vec3<_ValType>(left.y * right.z - left.z * right.y,
                              left.z * right.x - left.x * right.z,
                              left.x * right.y - left.y * right.x);
    }

    template<typename _ValType>
    Vec3<_ValType> doNormalize(const Vec3<_ValType>& vec)
    {
        double length = vec.getLength();
        if(length == 0.0)
        {
            throw std::runtime_error("Length of vector is 0!");
        }
        return Vec3<_ValType>((_ValType)((double)vec.x / length),
                              (_ValType)((double)vec.y / length),
                              (_ValType)((double)vec.z / length));
    }

	template<typename _ValType>
	Vec3<_ValType> componentWiseAbs(const Vec3<_ValType>& vec)
	{
		return Vec3<_ValType>(std::abs(vec.x),
							  std::abs(vec.y),
							  std::abs(vec.z));
	}

    template<typename _ValType>
    Vec3<_ValType> componentWiseMin(const Vec3<_ValType>& left, const Vec3<_ValType>& right)
    {
        return Vec3<_ValType>(std::min(left.x, right.x),
                              std::min(left.y, right.y),
                              std::min(left.z, right.z));
    }

	template<typename _ValType>
	Vec3<_ValType> componentWiseMax(const Vec3<_ValType>& left, const Vec3<_ValType>& right)
	{
		return Vec3<_ValType>(std::max(left.x, right.x),
			std::max(left.y, right.y),
			std::max(left.z, right.z));
	}

	template<typename _ValType>
	_ValType maxComponent(const Vec3<_ValType>& vec)
	{
		return std::max(vec.x, std::max(vec.y, vec.z));
	}

	template<typename _ValType>
	_ValType minComponent(const Vec3<_ValType>& vec)
	{
		return std::min(vec.x, std::min(vec.y, vec.z));
	}

    template<typename _ValType>
    Vec3<_ValType> clamp(const Vec3<_ValType>& v, const Vec3<_ValType>& min, const Vec3<_ValType>& max)
    {
        return Vec3<_ValType>(std::max(min.x, std::min(v.x, max.x)),
                              std::max(min.y, std::min(v.y, max.y)), 
                              std::max(min.z, std::min(v.z, max.z)));
    }

    template<typename _ValType>
    Vec3<_ValType> operator*(_ValType left, const Vec3<_ValType>& right)
    {
        return Vec3<_ValType>(left * right.x, left * right.y, left * right.z);
    }

    template<typename _ValType>
    std::ostream& operator<<(std::ostream& stream, const Vec3<_ValType>& vec)
    {
        stream << vec.x << " " << vec.y << " " << vec.z;
        return stream;
    }

    template<typename _ValType>
    std::istream& operator>>(std::istream& stream, Vec3<_ValType>& vec)
    {
        stream >> vec.x >> vec.y >> vec.z;
        return stream;
    }

    typedef Vec3<float> Vec3f;
    typedef Vec3<unsigned int> Vec3ui;
    typedef Vec3<int> Vec3i;
    typedef Vec3<double> Vec3d;
}