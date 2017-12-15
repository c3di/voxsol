#pragma once

#include <sstream>
#include <vector>

#include "Vec2.h"
#include "Vec3.h"

namespace libmmv
{
    template<typename _ValType>
    class Matrix3x3
    {
    public:
        Matrix3x3() {};

        Matrix3x3(_ValType c0x, _ValType c0y, _ValType c0z, _ValType c1x, _ValType c1y, _ValType c1z, _ValType c2x, _ValType c2y, _ValType c2z) {
            at(0, 0) = c0x;
            at(0, 1) = c0y;
            at(0, 2) = c0z;
            at(1, 0) = c1x;
            at(1, 1) = c1y;
            at(1, 2) = c1z;
            at(2, 0) = c2x;
            at(2, 1) = c2y;
            at(2, 2) = c2z;
        }

        Matrix3x3(Vec3<_ValType> column0, Vec3<_ValType> column1, Vec3<_ValType> column2) {
            at(0, 0) = column0.x;
            at(0, 1) = column0.y;
            at(0, 2) = column0.z;
            at(1, 0) = column1.x;
            at(1, 1) = column1.y;
            at(1, 2) = column1.z;
            at(2, 0) = column2.x;
            at(2, 1) = column2.y;
            at(2, 2) = column2.z;
        }

        Matrix3x3(const std::vector<_ValType>& values) {
            if (values.size() != 9)
            {
                throw std::runtime_error("Cannot initialize 3x3 Matrix from vector of size other than 9!");
            }
            memcpy(this->values, &values[0], 9 * sizeof(_ValType));
        }

        Matrix3x3(const Matrix3x3& other) {
            memcpy(values, other.values, 9 * sizeof(_ValType));
        }

        Matrix3x3& operator=(const Matrix3x3& other) {
            memcpy(values, other.values, 9 * sizeof(_ValType));
            return *this;
        }

        Matrix3x3 inverse() {
            _ValType det = determinant();
            if (abs(det) == 0)
            {
                throw std::runtime_error("matrix not invertible");
            }
            _ValType oneOverDeterminant = static_cast<_ValType>(1.0) / det;

            Matrix3x3 inverted;
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    inverted.at(y, x) = ((x + y) % 2 == 1 ? -1 : 1) * determinantOfMinor(x, y) * oneOverDeterminant;
                }
            }
            return inverted;
        }

        _ValType determinant() {
            return (this->at(0, 0) * determinantOfMinor(0, 0))
                - (this->at(0, 1) * determinantOfMinor(0, 1))
                + (this->at(0, 2) * determinantOfMinor(0, 2));
        }

        const _ValType& at(unsigned int column, unsigned int row) const {
            return values[3 * column + row];
        }

        _ValType& at(unsigned int column, unsigned int row) {
            return values[3 * column + row];
        }

        bool operator==(const Matrix3x3& other) const {
            for (int i = 0; i < 9; i++)
            {
                if (values[i] != other.values[i])
                {
                    return false;
                }
            }
            return true;
        }

        static const Matrix3x3 identity;

        static Matrix3x3 translationMatrix(Vec2f& translate) {
            Matrix3x3 m = identity;
            m.at(2, 0) = translate.x;
            m.at(2, 1) = translate.y;
            return m;
        }

        static Matrix3x3 scaleMatrix(_ValType v) {
            Matrix3x3 m = identity;
            m.at(0, 0) = v;
            m.at(1, 1) = v;
            return m;
        }

        static Matrix3x3 rotationMatrix(_ValType phi) {
            const _ValType cphi = cos(phi);
            const _ValType sphi = sin(phi);
            Matrix3x3 m = Matrix3x3::identity;
            m.at(0, 0) = cphi;
            m.at(1, 0) = -sphi;
            m.at(0, 1) = sphi;
            m.at(1, 1) = cphi;
            return m;
        }

    protected:

        int indexOfMinor(int i, int droppedIndex) {
            if (i < droppedIndex)
                return i;
            return i + 1;
        }

        _ValType determinantOfMinor(int x, int y) {
            int r[2];
            for (int i = 0; i < 2; i++)
                r[i] = indexOfMinor(i, x);
            int c[2];
            for (int i = 0; i < 2; i++)
                c[i] = indexOfMinor(i, y);

            return at(r[0], c[0]) * at(r[1], c[1]) -
                at(r[0], c[1]) * at(r[1], c[0]);
        }

        _ValType values[9];
    };

    template<typename _ValType>
    bool operator!=(const Matrix3x3<_ValType>& one, const Matrix3x3<_ValType>& other) {
        return !(one == other);
    }

    template<typename _ValType>
    Vec2<_ValType> multiply(const Matrix3x3<_ValType>& matrix, const Vec2<_ValType>& vector) {
        Vec3<_ValType> v3(v.x, v.y, 1.0f);
        v3 = multiply(m, v3);
        return Vec2<_ValType>(v3.x / v3.z, v3.y / v3.z);
    }

    template<typename _ValType>
    Vec3<_ValType> multiply(const Matrix3x3<_ValType>& matrix, const Vec3<_ValType>& vector) {
        const _ValType x = m.at(0, 0)*v.x + m.at(1, 0)*v.y + m.at(2, 0)*v.z;
        const _ValType y = m.at(0, 1)*v.x + m.at(1, 1)*v.y + m.at(2, 1)*v.z;
        const _ValType z = m.at(0, 2)*v.x + m.at(1, 2)*v.y + m.at(2, 2)*v.z;
        return Vec3<_ValType>(x, y, z);
    }

    template<typename _ValType>
    Matrix3x3<_ValType> multiply(const Matrix3x3<_ValType>& a, const Matrix3x3<_ValType>& b) {
        Matrix3x3 m;
        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
            {
                m.at(i, j) = 0.0f;
                for (unsigned int k = 0; k < 3; k++)
                    m.at(i, j) += a.at(k, j)*b.at(i, k);
            }
        }
        return m;
    }

    template<typename _ValType>
    std::istream &operator >> (std::istream& is, Matrix3x3<_ValType>& value) {
        return is;
    }

    template<typename _ValType>
    std::ostream &operator<<(std::ostream& os, Matrix3x3<_ValType> value) {
        os << "(";
        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
            {
                os << value.at(i, j) << " ";
            }
            os << "| ";
        }
        os << ")";
        return os;
    }

    typedef Matrix3x3<float> Matrix3x3f;
    typedef Matrix3x3<double> Matrix3x3d;
}
