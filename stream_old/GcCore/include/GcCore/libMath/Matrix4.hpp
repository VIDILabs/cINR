#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

#include <GcCore/libMath/MathUtils.hpp>
#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace math
{
    /**
    * @brief 4-by-4 matrix class. It is a row-major matrix.
    *        The representation will be : 
    *        | L T |
    *        | 0 1 |
    * with L has the linear transformation and T the translation
    * transformation.
    */
    template<typename T>
    class Matrix4
    {
    public:

        /**
        * @brief Default constructor. It gives the identity matrix.
        */
        Matrix4()
        {
            _d.fill(0);
            for (uint32_t i = 0; i < 4; ++i) _d[i * 4 + i] = 1.0;
        }

        Matrix4(const T value)
        {
            for (auto it = _d.begin(); it != _d.end(); ++it) *it = value;
        }

        /**
        * @brief Access operator overload.
        */
        T& operator [] (const int i) { return _d[i]; }

        const T& operator [] (const int i) const { return _d[i]; }

        T & operator () (const int row, const int col) { return _d[row * 4 + col]; }

        const T & operator () (const int row, const int col) const { return _d[row * 4 + col]; }

        T* data() { return _d.data(); }

        const T* data() const { return _d.data(); }

        /**
        * @brief Overload of mathematics operator.
        */
        Matrix4<T>& operator -= (const Matrix4<T> &other)
        {
            for (uint32_t i = 0; i < 16; ++i) _d[i] -= other._d[i];
            return *this;
        }

        Matrix4<T>& operator += (const Matrix4<T> &other)
        {
            for (uint32_t i = 0; i < 16; ++i) _d[i] += other._d[i];
            return *this;
        }

        Matrix4<T>& operator *= (const Matrix4<T> &other)
        {
            return *this = *this * other;
        }

        Matrix4<T>& operator *= (const T value)
        {
            for (uint32_t i = 0; i < 16; ++i) _d[i] *= value;
            return *this;
        }

        Matrix4<T> operator - (const Matrix4<T> &other) const
        {
            return Matrix4(*this) -= other;
        }

        Matrix4<T> operator - () const
        {
            return *this * -1;
        }

        Matrix4<T> operator + (const Matrix4<T> &other) const
        {
            return Matrix4<T>(*this) += other;
        }

        Matrix4<T> operator * (const Matrix4<T> &other) const
        {
            Matrix4<T> result(0);
            for (uint32_t row = 0; row < 4; ++row)
                for (uint32_t col = 0; col < 4; ++col)
                    for (uint32_t k = 0; k < 4; ++k)
                        result(row, col) += (*this)(row, k) * other(k, col);
            return result;
        }

        Vector<T, 4> operator * (const Vector<T, 4> &vec) const
        {
            Vector<T, 4> result;
            for (uint32_t i = 0; i < 4; ++i)
                for (uint32_t j = 0; j < 4; ++j)
                    result[i] += (*this)(i, j) * vec[j];
            return result;
        }

        Matrix4<T> operator * (const T value) const { return Matrix4<T>(*this) *= value; }

        /**
        * @brief Overload equals operator.
        */
        bool operator == (const Matrix4<T> &other) const
        {
            for (uint32_t i = 0; i < 16; ++i)
            {
                if (!tdns::math::utils::almost_equal(_d[i], other._d[i], 4)) return false;
            }
            return true;
        }

        bool operator != (const Matrix4<T> &other) const { return !(*this == other); }

        /**
        * @brief Gives if the matrix is an affine transformation.
        *        It supposes the matrix is | L T |
        *                                  | 0 1 |
        * with last row is [0 0 0 1].
        *
        * @return true if yes, else false
        */
        bool is_affine() const
        {
            return std::abs(_d[15] - 1) + std::abs(_d[14])
                + std::abs(_d[13]) + std::abs(_d[12])
                < std::numeric_limits<double>::epsilon();
        }

        /**
        * \brief Gives the determinant of the matrix. It assumes
        * the last row of the matrix is [0 0 0 1].
        *
        * \return Determinant
        */
        T get_determinant() const
        {
            assert(is_affine());
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
                - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
                + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        }

        /**
        * \brief Gives the inverse of the matrix. It assumes
        * the last row of the matrix is [0 0 0 1] then :
        *
        *    M = | L T | => M^-1 = | L^-1 -L^-1*T |
        *        | 0 1 |           |  0      1    |
        *
        * \return Inverse of the matrix
        */
        Matrix4<T> get_inverse_matrix() const
        {
            assert(is_affine());

            T det = get_determinant();
            //non-singular check
            assert(std::abs(det) > std::numeric_limits<T>::epsilon());

            const Matrix4<T> &matrix = (*this); //easier to read
            Matrix4<T> inverse;
            T invDet = T(1) / det; // / cost more than *

                                        //do L^-1
            inverse(0, 0) = (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) * invDet;
            inverse(0, 1) = -(matrix(0, 1) * matrix(2, 2) - matrix(0, 2) * matrix(2, 1)) * invDet;
            inverse(0, 2) = (matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1)) * invDet;

            inverse(1, 0) = -(matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) * invDet;
            inverse(1, 1) = (matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0)) * invDet;
            inverse(1, 2) = -(matrix(0, 0) * matrix(1, 2) - matrix(0, 2) * matrix(1, 0)) * invDet;

            inverse(2, 0) = (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0)) * invDet;
            inverse(2, 1) = -(matrix(0, 0) * matrix(2, 1) - matrix(0, 1) * matrix(2, 0)) * invDet;
            inverse(2, 2) = (matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0)) * invDet;

            //then get -L^-1 * T for the traslation part
            for (uint32_t i = 0; i < 3; ++i)
            {
                for (uint32_t j = 0; j < 3; ++j)
                {
                    inverse(i, 3) -= matrix(j, 3) * inverse(i, j);
                }
            }

            return inverse;
        }

        /**
        * \brief Gives the inverse of the matrix using Gauss methode.
        * /!\ error in precision need to check that
        * \return Inverse of the matrix
        */
        /*Matrix4 gauss()
        {
        double det = get_determinant();
        //non-singular check
        assert(std::abs(det) > std::numeric_limits<double>::epsilon());

        Matrix4 inverse;
        Matrix4 matrix(*this);
        for(uint32_t i = 0; i < 4; ++i)
        {
        //check if the coef is not null
        if(utils::almost_equal(matrix(i, i), 0.0, 2))
        {
        //swap line with other row ?
        bool done = false;
        for(uint32_t row = 0; row < 4; ++row)
        {
        if(row != i && !utils::almost_equal(matrix(row, i), 0.0, 2))
        {
        //swap row
        for(uint32_t index = 0; index < 4; ++index)
        {
        std::swap(matrix(i, index), matrix(row, index));
        std::swap(inverse(i, index), inverse(row, index));
        }

        done = true;
        break;
        }
        }

        //swap line with other column ?
        if(!done)
        {
        for(uint32_t col = 0; col < 4; ++col)
        {
        if(col != i && !utils::almost_equal(matrix(i, col), 0.0, 2))
        {
        //swap row
        for(uint32_t index = 0; index < 4; ++index)
        {
        std::swap(matrix(index, i), matrix(index, col));
        std::swap(inverse(index, i), inverse(index, col));
        }

        done = true;
        break;
        }
        }
        }
        assert(done);
        }//end if matrix(i, i) == 0.0

        //devided the by get the 1 in the diagonal
        double divider = 1.0 / matrix(i, i);
        for(uint32_t j = 0; j < 4; ++j)
        {
        matrix(i, j) *= divider;
        inverse(i, j) *= divider;
        }

        //then get 0 in the column i
        double factor;
        for(uint32_t row = 0; row < 4; ++row)
        {
        if(i != row)
        {
        factor = matrix(i, i) * matrix(row, i);
        for(uint32_t col = 0; col < 4; ++col)
        {
        matrix(row, col) -= factor * matrix(i, col);
        inverse(row, col) -= factor * inverse(i, col);
        }

        }
        }
        }

        return inverse;
        }*/

        void translate(const Vector<T, 3> &vec)
        {
            _d[3] += vec[0];
            _d[7] += vec[1];
            _d[11] += vec[2];
        }


        /**
        * @brief Gives the transpose of the matrix.
        *
        * @return Transposed matrix
        */
        Matrix4<T> get_transpose() const
        {
            Matrix4<T> transpose;
            for (uint32_t i = 0; i < 4; ++i)
            {
                for (uint32_t j = 0; j < 4; ++j)
                {
                    transpose(i, j) = (*this)(j, i);
                }
            }
            return transpose;
        }

        /**
        * @brief Gives a matrix with only the translation transformation (T)
        * of the current matrix.
        *
        * @return Translation matrix
        */
        Matrix4<T> get_translation_factor() const
        {
            Matrix4<T> matrix;
            matrix[3] = _d[3];
            matrix[7] = _d[7];
            matrix[11] = _d[11];
            return matrix;
        }

        /**
        * @brief Gives a matrix with only the linear transformation (L)
        * of the current matrix.
        *
        * @return Linear matrix
        */
        Matrix4<T> get_linear_factor() const
        {
            Matrix4<T> matrix;
            matrix[0] = _d[0]; matrix[1] = _d[1]; matrix[2] = _d[2];
            matrix[4] = _d[4]; matrix[5] = _d[5]; matrix[6] = _d[6];
            matrix[8] = _d[8]; matrix[9] = _d[9]; matrix[10] = _d[10];
            return matrix;
        }

    private:
        /**
        * Member data
        */
        std::array<T, 16> _d;
    };

    typedef Matrix4<double> Matrix4d;
    typedef Matrix4<float>  Matrix4f;
} //namespace math
} //namespace tdns