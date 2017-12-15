#pragma once
#include <string>
#include <algorithm>
#include <sstream>

namespace libmmv
{
	template<typename _ValType>
	class Vec2
	{
	public:
		union {
			struct
			{
				_ValType x;
				_ValType y;
			};
			_ValType dim[2];
		};

		inline int nDim() { return 2; };

		Vec2() {};
		Vec2(_ValType x, _ValType y) : x(x), y(y) {};
		Vec2(const Vec2<_ValType>& xy) : x(xy.x), y(xy.y) {};
		~Vec2() {};

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
			return std::to_string( x ) + "/" + std::to_string( y );
		};

		Vec2 makeInverse() const
		{
			double x = 1.0 / (double)this->x;
			double y = 1.0 / (double)this->y;
			return Vec2((_ValType)x, (_ValType)y);
		};

		const _ValType& operator[](unsigned int index) const { return dim[index]; };
		_ValType& operator[](unsigned int index) { return dim[index]; };

		Vec2& operator+=(const Vec2& other)
		{
			x += other.x;
			y += other.y;
			return *this;
		};

		Vec2& operator-=(const Vec2& other)
		{
			x -= other.x;
			y -= other.y;
			return *this;
		};

		Vec2& operator*=(const Vec2& other)
		{
			x *= other.x;
			y *= other.y;
			return *this;
		};

		Vec2& operator/=(const Vec2& other)
		{
			x /= other.x;
			y /= other.y;
			return *this;
		};

		Vec2& operator*=(_ValType scalar)
		{
			x *= scalar;
			y *= scalar;
			return *this;
		};

		Vec2& operator/=(_ValType scalar)
		{
			x /= scalar;
			y /= scalar;
			return *this;
		};

		Vec2 operator+(const Vec2& other) const
		{
			return Vec2(x + other.x, y + other.y);
		};

		Vec2 operator-(const Vec2& other) const
		{
			return Vec2(x - other.x, y - other.y);
		};

		Vec2 operator*(const Vec2& other) const
		{
			return Vec2(x * other.x, y * other.y);
		};

		Vec2 operator/(const Vec2& other) const
		{
			return Vec2(x / other.x, y / other.y);
		};

		Vec2 operator-() const
		{
			return Vec2(-x, -y);
		};

		Vec2 operator*(_ValType scalar) const
		{
			return Vec2(x * scalar, y * scalar);
		};

		Vec2 operator/(_ValType scalar) const
		{
			return Vec2(x / scalar, y / scalar);
		};

		bool operator==(const Vec2& other) const
		{
			return x == other.x && y == other.y;
		};

		bool operator!=(const Vec2& other) const
		{
			return x != other.x || y != other.y;
		}

		template<typename _CastValue>
		operator Vec2<_CastValue>() const
		{
			return Vec2<_CastValue>((_CastValue)x, (_CastValue)y);
		}

		// the following compare functions assume coordinate priorities z,y,x
		bool operator<(const Vec2& other) const
		{
			if (y < other.y)
				return true;
			if (y > other.y)
				return false;
			if (x < other.x)
				return true;
			return false;
		};

		bool operator<=(const Vec2& other) const
		{
			if (y < other.y)
				return true;
			if (y > other.y)
				return false;
			if (x <= other.x)
				return true;
			return false;
		};

		bool operator>(const Vec2& other) const
		{
			if (y > other.y)
				return true;
			if (y < other.y)
				return false;
			if (x > other.x)
				return true;
			return false;
		};

		bool operator>=(const Vec2& other) const
		{
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
	_ValType dotProduct(const Vec2<_ValType>& left, const Vec2<_ValType>& right)
	{
		return left.x * right.x + left.y * right.y;
	}

	template<typename _ValType>
	Vec2<_ValType> doNormalize(const Vec2<_ValType>& vec)
	{
		double length = vec.getLength();
		if (length == 0.0)
		{
			throw std::runtime_error("Length of vector is 0!");
		}
		return Vec2<_ValType>((_ValType)(
			(double)vec.x / length),
			(_ValType)((double)vec.y / length));
	}

	template<typename _ValType>
	Vec2<_ValType> componentWiseAbs(const Vec2<_ValType>& vec)
	{
		return Vec2<_ValType>(std::abs(vec.x),
							  std::abs(vec.y));
	}

	template<typename _ValType>
	Vec2<_ValType> componentWiseMin(const Vec2<_ValType>& left, const Vec2<_ValType>& right)
	{
		return Vec2<_ValType>(
			std::min(left.x, right.x),
			std::min(left.y, right.y));
	}

	template<typename _ValType>
	Vec2<_ValType> componentWiseMax(const Vec2<_ValType>& left, const Vec2<_ValType>& right)
	{
		return Vec2<_ValType>(std::max(left.x, right.x),
			std::max(left.y, right.y));
	}

	template<typename _ValType>
	_ValType maxComponent(const Vec2<_ValType>& vec)
	{
		return std::max(vec.x, vec.y);
	}

	template<typename _ValType>
	_ValType minComponent(const Vec2<_ValType>& vec)
	{
		return std::min(vec.x, vec.y);
	}


	template<typename _ValType>
	Vec2<_ValType> clamp(const Vec2<_ValType>& v, const Vec2<_ValType>& min, const Vec2<_ValType>& max)
	{
		return Vec2<_ValType>(std::max(min.x, std::min(v.x, max.x)),
							  std::max(min.y, std::min(v.y, max.y)) );
	}

	template<typename _ValType>
	Vec2<_ValType> operator*(_ValType left, const Vec2<_ValType>& right)
	{
		return Vec2<_ValType>(left * right.x, left * right.y);
	}

	template<typename _ValType>
	std::ostream& operator<<(std::ostream& stream, const Vec2<_ValType>& vec)
	{
		stream << vec.x << " " << vec.y;
		return stream;
	}

	template<typename _ValType>
	std::istream& operator >> (std::istream& stream, Vec2<_ValType>& vec)
	{
		stream >> vec.x >> vec.y;
		return stream;
	}

	typedef Vec2<float> Vec2f;
	typedef Vec2<unsigned int> Vec2ui;
	typedef Vec2<int> Vec2i;
	typedef Vec2<double> Vec2d;
}