#pragma once

// #include "framework/error/Exception.h"
#include "Vec2.h"
#include "Vec3.h"

namespace ettention
{
	template<typename _ValType>
	class Vec4
	{
	public:
		union {
			struct
			{
				_ValType x;
				_ValType y;
				_ValType z;
				_ValType w;
			};
			_ValType dim[4];
		};

		Vec4() {};
		Vec4(_ValType x, _ValType y, _ValType z, _ValType w) : x(x), y(y), z(z), w(w) {};
		Vec4(const Vec3<_ValType>& xyz, _ValType w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {};
		~Vec4() {};

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
			return std::to_string(x) + "/" + std::to_string(y) + "/" + std::to_string(z) + "/" + std::to_string(w);
		};

		Vec4 makeInverse() const
		{
			double x = 1.0 / (double)this->x;
			double y = 1.0 / (double)this->y;
			double z = 1.0 / (double)this->z;
			double w = 1.0 / (double)this->z;
			return Vec4((_ValType)x, (_ValType)y, (_ValType)z, (_ValType)w);
		};

		Vec2<_ValType> xy() const { return Vec2<_ValType>(x, y); };
		Vec2<_ValType> yz() const { return Vec2<_ValType>(y, z); };
		Vec2<_ValType> xz() const { return Vec2<_ValType>(x, z); };
		Vec3<_ValType> xyz() const { return Vec3<_ValType>(x, y, z); };

		const _ValType& operator[](unsigned int index) const { return dim[index]; };
		_ValType& operator[](unsigned int index) { return dim[index]; };

		Vec4& operator+=(const Vec4& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
			w += other.w;
			return *this;
		};

		Vec4& operator-=(const Vec4& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
			w -= other.w;
			return *this;
		};

		Vec4& operator*=(const Vec4& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;
			w *= other.w;
			return *this;
		};

		Vec4& operator/=(const Vec4& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;
			w /= other.w;
			return *this;
		};

		Vec4& operator*=(_ValType scalar)
		{
			x *= scalar;
			y *= scalar;
			z *= scalar;
			w *= scalar;
			return *this;
		};

		Vec4& operator/=(_ValType scalar)
		{
			x /= scalar;
			y /= scalar;
			z /= scalar;
			w /= scalar;
			return *this;
		};

		Vec4 operator+(const Vec4& other) const
		{
			return Vec4(x + other.x, y + other.y, z + other.z, w + other.w );
		};

		Vec4 operator-(const Vec4& other) const
		{
			return Vec4(x - other.x, y - other.y, z - other.z, w - other.w );
		};

		Vec4 operator*(const Vec4& other) const
		{
			return Vec4(x * other.x, y * other.y, z * other.z, w * other.w );
		};

		Vec4 operator/(const Vec4& other) const
		{
			return Vec4(x / other.x, y / other.y, z / other.z, w / other.w );
		};

		Vec4 operator-() const
		{
			return Vec4(-x, -y, -z, -w);
		};

		Vec4 operator*(_ValType scalar) const
		{
			return Vec4(x * scalar, y * scalar, z * scalar, w * scalar );
		};

		Vec4 operator/(_ValType scalar) const
		{
			return Vec4(x / scalar, y / scalar, z / scalar, w / scalar );
		};

		bool operator==(const Vec4& other) const
		{
			return x == other.x && y == other.y && z == other.z && w == other.w;
		};

		bool operator!=(const Vec4& other) const
		{
			return x != other.x || y != other.y || z != other.z || w != other.w;
		}

		template<typename _CastValue>
		operator Vec4<_CastValue>() const
		{
			return Vec4<_CastValue>((_CastValue)x, (_CastValue)y, (_CastValue)z, (_CastValue)w );
		}

		// the following compare functions assume coordinate priorities w,z,y,x
		bool operator<(const Vec4& other) const
		{
			if (w < other.w)
				return true;
			if (w > other.w)
				return false;
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

		bool operator<=(const Vec4& other) const
		{
			if (w < other.w)
				return true;
			if (w > other.w)
				return false;
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

		bool operator>(const Vec4& other) const
		{
			if (w > other.w)
				return true;
			if (w < other.w)
				return false;
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

		bool operator>=(const Vec4& other) const
		{
			if (w > other.w)
				return true;
			if (w < other.w)
				return false;
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
	_ValType dotProduct(const Vec4<_ValType>& left, const Vec4<_ValType>& right)
	{
		return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
	}

	template<typename _ValType>
	Vec4<_ValType> doNormalize(const Vec4<_ValType>& vec)
	{
		double length = vec.getLength();
		if (length == 0.0)
		{
			throw std::runtime_error("Length of vector is 0!");
		}
		return Vec4<_ValType>((_ValType)((double)vec.x / length),
							  (_ValType)((double)vec.y / length),
				    	      (_ValType)((double)vec.z / length),
							  (_ValType)((double)vec.w / length) );
	}

	template<typename _ValType>
	Vec4<_ValType> componentWiseAbs(const Vec4<_ValType>& vec)
	{
		return Vec4<_ValType>(std::abs(vec.x),
							  std::abs(vec.y),
							  std::abs(vec.z),
							  std::abs(vec.w));
	}

	template<typename _ValType>
	Vec4<_ValType> componentWiseMin(const Vec4<_ValType>& left, const Vec4<_ValType>& right)
	{
		return Vec4<_ValType>(
			std::min(left.x, right.x),
			std::min(left.y, right.y),
			std::min(left.z, right.z),
			std::min(left.w, right.w));
	}

	template<typename _ValType>
	Vec4<_ValType> componentWiseMax(const Vec4<_ValType>& left, const Vec4<_ValType>& right)
	{
		return Vec4<_ValType>(std::max(left.x, right.x),
			std::max(left.y, right.y),
			std::max(left.z, right.z),
			std::max(left.w, right.w) );
	}

	template<typename _ValType>
	_ValType maxComponent(const Vec4<_ValType>& vec)
	{
		return std::max(vec.x, std::max(vec.y, std::max(vec.z, vec.w) ) );
	}

	template<typename _ValType>
	_ValType minComponent(const Vec4<_ValType>& vec)
	{
		return std::min(vec.x, std::min(vec.y, std::min(vec.z, vec.w)));
	}

	template<typename _ValType>
	Vec4<_ValType> clamp(const Vec4<_ValType>& v, const Vec4<_ValType>& min, const Vec4<_ValType>& max)
	{
		return Vec4<_ValType>(std::max(min.x, std::min(v.x, max.x)),
			std::max(min.y, std::min(v.y, max.y)),
			std::max(min.z, std::min(v.z, max.z)),
			std::max(min.w, std::min(v.w, max.w)) );
	}

	template<typename _ValType>
	Vec4<_ValType> operator*(_ValType left, const Vec4<_ValType>& right)
	{
		return Vec3<_ValType>(left * right.x, left * right.y, left * right.z, left * right.w );
	}

	template<typename _ValType>
	std::ostream& operator<<(std::ostream& stream, const Vec4<_ValType>& vec)
	{
		stream << vec.x << " " << vec.y << " " << vec.z << " " << vec.w;
		return stream;
	}

	template<typename _ValType>
	std::istream& operator >> (std::istream& stream, Vec4<_ValType>& vec)
	{
		stream >> vec.x >> vec.y >> vec.z >> vec.w;
		return stream;
	}

	typedef Vec4<float> Vec4f;
	typedef Vec4<unsigned int> Vec4ui;
	typedef Vec4<int> Vec4i;
	typedef Vec4<double> Vec4d;
}