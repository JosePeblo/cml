#pragma once
#include <iostream>
#define CMTLOG(x) std::cout << x

#include <math.h>
#include <limits>

#define PI 3.14159265358979323846

namespace cml {

struct mat3 {
    float matrix[3][3] = { 0 };

    mat3 operator * (const mat3 &m) const {
        mat3 temp;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                temp.matrix[i][j] = matrix[i][0] * m.matrix[0][j] +
                                    matrix[i][1] * m.matrix[1][j] +
                                    matrix[i][2] * m.matrix[2][j];
            }
        }
        return temp;
    }
    void PRINT() const {
        CMTLOG(matrix[0][0] << ',' << matrix[0][1] << ',' << matrix[0][2] << '\n' <<
               matrix[1][0] << ',' << matrix[1][1] << ',' << matrix[1][2] << '\n' <<
               matrix[2][0] << ',' << matrix[2][1] << ',' << matrix[2][2] << '\n');
    }
};

struct mat4 {
    float matrix[4][4] = { 0 };
    
    mat4 operator * (const mat4 &m) const {
        mat4 temp;
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                temp.matrix[i][j] = matrix[i][0] * m.matrix[0][j] +
                                    matrix[i][1] * m.matrix[1][j] +
                                    matrix[i][2] * m.matrix[2][j] +
                                    matrix[i][3] * m.matrix[3][j];
            }
        }
        return temp;
    }

    float ( & operator [] (size_t i)) [4] {
        return matrix[i];
    }

    void PRINT() const {
        CMTLOG(matrix[0][0] << ',' << matrix[0][1] << ',' << matrix[0][2] << ',' << matrix[0][3] << '\n' <<
               matrix[1][0] << ',' << matrix[1][1] << ',' << matrix[1][2] << ',' << matrix[1][3] << '\n' <<
               matrix[2][0] << ',' << matrix[2][1] << ',' << matrix[2][2] << ',' << matrix[2][3] << '\n' <<
               matrix[3][0] << ',' << matrix[3][1] << ',' << matrix[3][2] << ',' << matrix[3][3] << '\n');
    }
};

struct vec2 {
    float x = 0.f;
    float y = 0.f;

    vec2(): x(0.f), y(0.f) {}
    vec2(float x, float y): x(x), y(y) {}
    
    // Linear algebra
    float dot(const vec2 &v) const {
        return x * v.x + y * v.y;
    }

    float mag() const {
        return sqrtf(x * x + y * y);
    }

    vec2 norm() {
        return *this/mag();
    }

    // Math functions with other vectors 
    vec2 operator + (const vec2 &v) const {
        return vec2(x + v.x, y + v.y);
    }

    vec2 operator - (const vec2 &v) const {
        return vec2(x - v.x, y - v.y);
    }

    vec2 operator * (const vec2 &v) const {
        return vec2(x * v.x, y * v.y);
    }

    vec2 operator / (const vec2 &v) const {
        return vec2(x / v.x, y / v.y);
    }

    vec2& operator = (const vec2 &v) {
        x = v.x;
        y = v.y;
        return *this;
    }

    vec2& operator += (const vec2 &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    vec2& operator -= (const vec2 &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    vec2& operator *= (const vec2 &v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    vec2& operator /= (const vec2 &v) {
        x /= v.x;
        y /= v.y;
        return *this;
    }

    // Boolean operators
    bool operator < (const vec2 &v) const {
        return (this->mag() < v.mag()) ? true : false;
    }

    bool operator > (const vec2 &v) const {
        return (this->mag() > v.mag()) ? true : false;
    }

    bool operator <= (const vec2 &v) const {
        return (this->mag() <= v.mag()) ? true : false;
    }

    bool operator >= (const vec2 &v) const {
        return (this->mag() >= v.mag()) ? true : false;
    }

    bool operator == (const vec2 &v) const {
        return (x == v.x && y == v.y) ? true : false;
    }

    bool operator != (const vec2 &v) const {
        return (x != v.x || y != v.y) ? true : false;
    }

    // Math functions with floats
    vec2 operator + (const float &f) const {
        return vec2(x + f, y + f);
    }

    vec2 operator - (const float &f) const {
        return vec2(x - f, y - f);
    }

    vec2 operator * (const float &f) const {
        return vec2(x * f, y * f);
    }

    vec2 operator / (const float &f) const {
        return vec2(x / f, y / f);
    }

    vec2& operator += (const float &f) {
        x += f;
        y += f;
        return *this;
    }

    vec2& operator -= (const float &f) {
        x -= f;
        y -= f;
        return *this;
    }

    vec2& operator *= (const float &f) {
        x *= f;
        y *= f;
        return *this;
    }

    vec2& operator /= (const float &f) {
        x /= f;
        y /= f;
        return *this;
    }

    void PRINT() const {
        CMTLOG(x << ',' << y << '\n');
    }
};

struct ivec2 {
    int x = 0;
    int y = 0;

    ivec2(): x(0), y(0) {}
    ivec2(int x, int y): x(x), y(y) {}
    
    // Linear algebra

    float dot(const ivec2 &v) const {
        return x * v.x + y * v.y;
    }

    float mag() const {
        return sqrtf(x * x + y * y);
    }

    ivec2 norm() {
        return *this/mag();
    }

    // Math functions with other vectors 

    ivec2 operator + (const ivec2 &v) const {
        return ivec2(x + v.x, y + v.y);
    }

    ivec2 operator - (const ivec2 &v) const {
        return ivec2(x - v.x, y - v.y);
    }

    ivec2 operator * (const ivec2 &v) const {
        return ivec2(x * v.x, y * v.y);
    }

    ivec2 operator / (const ivec2 &v) const {
        return ivec2(x / v.x, y / v.y);
    }

    ivec2& operator = (const ivec2 &v) {
        x = v.x;
        y = v.y;
        return *this;
    }

    ivec2& operator += (const ivec2 &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    ivec2& operator -= (const ivec2 &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    ivec2& operator *= (const ivec2 &v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    ivec2& operator /= (const ivec2 &v) {
        x /= v.x;
        y /= v.y;
        return *this;
    }

    // Boolean operators
    bool operator < (const ivec2 &v) const {
        return (this->mag() < v.mag()) ? true : false;
    }

    bool operator > (const ivec2 &v) const {
        return (this->mag() > v.mag()) ? true : false;
    }

    bool operator <= (const ivec2 &v) const {
        return (this->mag() <= v.mag()) ? true : false;
    }

    bool operator >= (const ivec2 &v) const {
        return (this->mag() >= v.mag()) ? true : false;
    }

    bool operator == (const ivec2 &v) const {
        return (x == v.x && y == v.y) ? true : false;
    }

    bool operator != (const ivec2 &v) const {
        return (x != v.x || y != v.y) ? true : false;
    }

    // Math functions with floats
    ivec2 operator + (const float &f) const {
        return ivec2(x + f, y + f);
    }

    ivec2 operator - (const float &f) const {
        return ivec2(x - f, y - f);
    }

    ivec2 operator * (const float &f) const {
        return ivec2(x * f, y * f);
    }

    ivec2 operator / (const float &f) const {
        return ivec2(x / f, y / f);
    }

    ivec2& operator += (const float &f) {
        x += f;
        y += f;
        return *this;
    }

    ivec2& operator -= (const float &f) {
        x -= f;
        y -= f;
        return *this;
    }

    ivec2& operator *= (const float &f) {
        x *= f;
        y *= f;
        return *this;
    }

    ivec2& operator /= (const float &f) {
        x /= f;
        y /= f;
        return *this;
    }

    void PRINT() const {
        CMTLOG(x << ',' << y << '\n');
    }
};

struct vec3 {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    vec3(): x(0.f), y(0.f), z(0.f){}
    vec3(float x, float y, float z): x(x), y(y), z(z){}
    
    // Linear algebra fuctions
    vec3 operator * (const mat3 &m) const {
        return vec3(x * m.matrix[0][0] + y * m.matrix[1][0] + z * m.matrix[2][0],
                    x * m.matrix[0][1] + y * m.matrix[1][1] + z * m.matrix[2][1],
                    x * m.matrix[0][2] + y * m.matrix[1][2] + z * m.matrix[2][2]);
    }

    vec3& operator *= (const mat3 &m) {
        vec3 v = *this * m;
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        return *this;
    }

    // Cheeky math functions with dirty asumptions HAVE FEAR
    vec3 operator * (const mat4 &m) const {
        vec3 out(x * m.matrix[0][0] + y * m.matrix[1][0] + z * m.matrix[2][0] + m.matrix[3][0],
                 x * m.matrix[0][1] + y * m.matrix[1][1] + z * m.matrix[2][1] + m.matrix[3][1],
                 x * m.matrix[0][2] + y * m.matrix[1][2] + z * m.matrix[2][2] + m.matrix[3][2]);

        float w = x * m.matrix[0][3] + y * m.matrix[1][3] + z * m.matrix[2][3] + m.matrix[3][3];

        if(w != 0.0f)
        {
            out /= w;
        }
        return out;
    }

    vec3& operator *= (const mat4 &m) {
        vec3 v = *this * m;
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        return *this;
    }
    // End of the butchery

    float dot(const vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    vec3 cross(const vec3 &v) const {
        return vec3(y * v.z - v.y * z, 
                    z * v.x - v.z * x,
                    x * v.y - v.x * y); 
    }
    
    float mag() const {
        return sqrtf(x * x + y * y + z * z);
    }

    vec3 norm() {
        return *this/mag();
    }

    // Math functions with other vectors 
    vec3 operator + (const vec3 &v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }

    vec3 operator - (const vec3 &v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    vec3 operator * (const vec3 &v) const {
        return vec3(x * v.x, y * v.y, z * v.z);
    }

    vec3 operator / (const vec3 &v) const {
        return vec3(x / v.x, y / v.y, z / v.z);
    }

    vec3& operator = (const vec3 &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    vec3& operator += (const vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    vec3& operator -= (const vec3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    vec3& operator *= (const vec3 &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    vec3& operator /= (const vec3 &v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    // Boolean operators
    bool operator < (const vec3 &v) const {
        return (this->mag() < v.mag()) ? true : false;
    }

    bool operator > (const vec3 &v) const {
        return (this->mag() > v.mag()) ? true : false;
    }

    bool operator <= (const vec3 &v) const {
        return (this->mag() <= v.mag()) ? true : false;
    }

    bool operator >= (const vec3 &v) const {
        return (this->mag() >= v.mag()) ? true : false;
    }

    bool operator == (const vec3 &v) const {
        return (x == v.x && y == v.y && z == v.z) ? true : false;
    }

    bool operator != (const vec3 &v) const {
        return (x != v.x || y != v.y || z != v.z) ? true : false;
    }

    // Math functions with floats
    vec3 operator + (const float &f) const {
        return vec3(x + f, y + f, z + f);
    }

    vec3 operator - (const float &f) const {
        return vec3(x - f, y - f, z - f);
    }

    vec3 operator * (const float &f) const {
        return vec3(x * f, y * f, z * f);
    }

    vec3 operator / (const float &f) const {
        return vec3(x / f, y / f, z / f);
    }

    vec3& operator += (const float &f) {
        x += f;
        y += f;
        z += f;
        return *this;
    }

    vec3& operator -= (const float &f) {
        x -= f;
        y -= f;
        z -= f;
        return *this;
    }

    vec3& operator *= (const float &f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    vec3& operator /= (const float &f) {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    vec3& negate() {
        return (*this) *= -1;
    }

    void PRINT() const {
        CMTLOG(x << ',' << y << ','<< z << '\n');
    }
};

struct vec4 
{
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float w = 0.f;

    vec4(): x(0.f), y(0.f), z(0.f), w(0.f){}
    vec4(float x, float y, float z, float w): x(x), y(y), z(z), w(w){}

    vec4 operator * (const mat4 &m) const {
        return vec4(x * m.matrix[0][0] + y * m.matrix[1][0] + z * m.matrix[2][0] + w * m.matrix[3][0],
                    x * m.matrix[0][1] + y * m.matrix[1][1] + z * m.matrix[2][1] + w * m.matrix[3][1],
                    x * m.matrix[0][2] + y * m.matrix[1][2] + z * m.matrix[2][2] + w * m.matrix[3][2],
                    x * m.matrix[0][3] + y * m.matrix[1][3] + z * m.matrix[2][3] + w * m.matrix[3][3]);
    }

    vec4& operator *= (const mat4 &m) {
        vec4 v = *this * m;
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        this->w = v.w;
        return *this;
    }

    // Linear algebra fuctions

    float dot(const vec4 &v) const {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    float mag() const {
        return sqrtf(x * x + y * y + z * z + w * w);
    }

    vec4 norm() {
        return *this/mag();
    }

    // Math functions with other vectors 
    vec4 operator + (const vec4 &v) const {
        return vec4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    vec4 operator - (const vec4 &v) const {
        return vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    vec4 operator * (const vec4 &v) const {
        return vec4(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    vec4 operator / (const vec4 &v) const {
        return vec4(x / v.x, y / v.y, z / v.z, w / v.w);
    }

    vec4& operator = (const vec4 &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    vec4& operator += (const vec4 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    vec4& operator -= (const vec4 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    vec4& operator *= (const vec4 &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }

    vec4& operator /= (const vec4 &v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        w /= v.w;
        return *this;
    }

    // Boolean operators
    bool operator < (const vec4 &v) const {
        return (this->mag() < v.mag()) ? true : false;
    }

    bool operator > (const vec4 &v) const {
        return (this->mag() > v.mag()) ? true : false;
    }

    bool operator <= (const vec4 &v) const {
        return (this->mag() <= v.mag()) ? true : false;
    }

    bool operator >= (const vec4 &v) const {
        return (this->mag() >= v.mag()) ? true : false;
    }

    bool operator == (const vec4 &v) const {
        return (x == v.x && y == v.y && z == v.z && w == v.w) ? true : false;
    }

    bool operator != (const vec4 &v) const {
        return (x != v.x || y != v.y || z != v.z || w != v.w) ? true : false;
    }

    // Math functions with floats
    vec4 operator + (const float &f) const {
        return vec4(x + f, y + f, z + f, w + f);
    }

    vec4 operator - (const float &f) const {
        return vec4(x - f, y - f, z - f, w - f);
    }

    vec4 operator * (const float &f) const {
        return vec4(x * f, y * f, z * f, w * f);
    }

    vec4 operator / (const float &f) const {
        return vec4(x / f, y / f, z / f, w / f);
    }

    vec4& operator += (const float &f) {
        x += f;
        y += f;
        z += f;
        w += f;
        return *this;
    }

    vec4& operator -= (const float &f) {
        x -= f;
        y -= f;
        z -= f;
        w -= f;
        return *this;
    }

    vec4& operator *= (const float &f) {
        x *= f;
        y *= f;
        z *= f;
        w *= f;
        return *this;
    }

    vec4& operator /= (const float &f) {
        x /= f;
        y /= f;
        z /= f;
        w /= f;
        return *this;
    }


    void PRINT() const {
        CMTLOG(x << ',' << y << ','<< z << ','<< w << '\n');
    }
};

struct quat {
    float x,y,z,w;

    // Quaternion operations
    // Implementations mostly from glm, trying to understand quaternions
    // https://github.com/g-truc/glm/blob/master/glm/detail/type_quat.inl
    quat operator*(float f) const {
        return { x*f, y*f, z*f, w*f };
    }

    quat operator+(const quat& that) const  {
        //(*this = detail::compute_quat_add<T, Q, detail::is_aligned<Q>::value>::call(*this, qua<T, Q>(q)));
        return { x + that.x, y + that.y, z + that.z, w + that.w };
    }

    quat operator/(float f) {
        return { x / f, y / f, z / f, w / f };
    }

    mat4 toMat4() const {
        /*
        q0 = w
        q1 = x
        q2 = y
        q3 = z
        */
        return {
            1 - 2 * y*y - 2 * z*z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0.0f,
            2 * x * y + 2 * w * z, 1 - 2 * x*x - 2 * z*z, 2 * y * z - 2 * w * x, 0.0f,
            2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x*x - 2 * y*y, 0.0f,
            0.0f                 , 0.0f                 , 0.0f                 , 1.0f
        };
    }
};


inline quat operator*(float f, const quat& q) {
    return q * f;
}

inline vec3 norm(const vec3& vec) {
    return vec/vec.mag();
}

inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y * v2.z - v2.y * v1.z, 
                v1.z * v2.x - v2.z * v1.x,
                v1.x * v2.y - v2.x * v1.y); 
}

inline float dot(const vec3& v1, const vec3& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

inline float dot(const quat& q1, const quat& q2) {
    return (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w);
}

inline mat4 ortho(float left, float right, float bottom, float top) {
    mat4 orth = {
        2/(right-left), 0, 0, -(right+left)/(right-left),
        0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom),
        0,     0,         -1,               0,
        0,     0,          0,               1
    };
    return orth;
}

inline mat4 ortho(float left, float right, float bottom, float top, float nearPlane, float farPlane) {
    mat4 orth = {
        2/(right-left), 0, 0, -(right+left)/(right-left),
        0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom),
        0, 0,  -2/(farPlane-nearPlane),  -(farPlane+nearPlane)/(farPlane-nearPlane),
        0, 0,       0,                     1
    };
    return orth;
}

inline mat4 perspective(float fov, float aspect, float nearPlane, float farPlane) {
    mat4 persp = {
        1/(aspect * tanf(fov/2)),        0,        0,        0,
        0,                         1/tanf(fov/2),  0,        0,
        0, 0, -(farPlane+nearPlane)/(farPlane-nearPlane), -(2*farPlane*nearPlane)/(farPlane-nearPlane),
        0, 0,            -1,                       0
    };
    return persp;
}

inline mat4 perspective(float fov, float width, float height, float nearPlane, float farPlane) {
    float aspect = height/width;
    return perspective(fov, aspect, nearPlane, farPlane);
}

inline mat4 lookAt(const vec3& look, const vec3& at, const vec3& up) {
    vec3 zAxis = norm(at - look);
    vec3 xAxis = norm(cross(zAxis, up));
    vec3 yAxis = cross(xAxis, zAxis);
    zAxis.negate();
    return {
        xAxis.x, xAxis.y, xAxis.z, -dot(xAxis, look),
        yAxis.x, yAxis.y, yAxis.z, -dot(yAxis, look),
        zAxis.x, zAxis.y, zAxis.z, -dot(zAxis, look),
           0,       0,       0,            1
    };
}

inline mat4 translate(const mat4& m, const vec3& v) {
    mat4 trans = m;
    trans.matrix[0][3] = v.x;
    trans.matrix[1][3] = v.y;
    trans.matrix[2][3] = v.z;
    return trans;
}

inline mat4 scale(const mat4& m, const vec3& v) {
    mat4 scal = m;
    scal.matrix[0][0] = v.x;
    scal.matrix[1][1] = v.y;
    scal.matrix[2][2] = v.z;
    return scal;
}

inline mat4 rotationX(float angle) {
    return {
        1,    0       ,     0       , 0,
        0, cosf(angle), -sinf(angle), 0,
        0, sinf(angle),  cosf(angle), 0,
        0,    0,            0,        1
    };
}

inline mat4 rotationY(float angle) {
    return {
        cosf(angle),  0, sinf(angle),  0,
            0,        1,     0,        0,
        -sinf(angle), 0, cosf(angle),  0,
            0,        0,     0,        1
    };    
}

inline mat4 rotationZ(float angle) {
    return {
        cosf(angle), -sinf(angle), 0, 0,
        sinf(angle),  cosf(angle), 0, 0,
            0      ,      0      , 1, 0,
            0      ,      0      , 0, 1
    };
}

inline mat3 rotationZ3(float angle) {
    return {
        cosf(angle), -sinf(angle), 0,
        sinf(angle),  cosf(angle), 0,
            0      ,      0      , 1
    };
}

const mat3 identity3 = {
    1,0,0,
    0,1,0,
    0,0,1
};

const mat4 identity4 = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
};

inline vec3 rotate (const vec3& vec, const float& angle, const vec3& normal) {
    mat4 m = cml::identity4;

    float a = angle;
    float c = cos(a);
    float s = sin(a);

    vec3 axis(norm(normal));
    vec3 temp(axis * (float(1) - c));

    mat4 rotate;
    rotate[0][0] = c + temp.x * axis.x;
    rotate[0][1] = temp.x * axis.y + s * axis.z;
    rotate[0][2] = temp.x * axis.z - s * axis.y;

    rotate[1][0] = temp.y * axis.x - s * axis.z;
    rotate[1][1] = c + temp.y * axis.y;
    rotate[1][2] = temp.y * axis.z + s * axis.x;

    rotate[2][0] = temp.z * axis.x + s * axis.y;
    rotate[2][1] = temp.z * axis.y - s * axis.x;
    rotate[2][2] = c + temp.z * axis.z;

    mat4 result;
    result[0][0] = m[0][0] * rotate[0][0] + m[1][0] * rotate[0][1] + m[2][0] * rotate[0][2];
    result[0][1] = m[0][1] * rotate[0][0] + m[1][1] * rotate[0][1] + m[2][1] * rotate[0][2];
    result[0][2] = m[0][2] * rotate[0][0] + m[1][2] * rotate[0][1] + m[2][2] * rotate[0][2];
    result[0][3] = m[0][3] * rotate[0][0] + m[1][3] * rotate[0][1] + m[2][3] * rotate[0][2];

    result[1][0] = m[0][0] * rotate[1][0] + m[1][0] * rotate[1][1] + m[2][0] * rotate[1][2];
    result[1][1] = m[0][1] * rotate[1][0] + m[1][1] * rotate[1][1] + m[2][1] * rotate[1][2];
    result[1][2] = m[0][2] * rotate[1][0] + m[1][2] * rotate[1][1] + m[2][2] * rotate[1][2];
    result[1][3] = m[0][3] * rotate[1][0] + m[1][3] * rotate[1][1] + m[2][3] * rotate[1][2];

    result[2][0] = m[0][0] * rotate[2][0] + m[1][0] * rotate[2][1] + m[2][0] * rotate[2][2];
    result[2][1] = m[0][1] * rotate[2][0] + m[1][1] * rotate[2][1] + m[2][1] * rotate[2][2];
    result[2][2] = m[0][2] * rotate[2][0] + m[1][2] * rotate[2][1] + m[2][2] * rotate[2][2];
    result[2][3] = m[0][3] * rotate[2][0] + m[1][3] * rotate[2][1] + m[2][3] * rotate[2][2];

    result[3][0] = m[3][0];
    result[3][1] = m[3][1];
    result[3][2] = m[3][2];
    result[3][3] = m[3][3];


    mat3 mult = {
        result[0][0], result[0][1], result[0][2],
        result[1][0], result[1][1], result[1][2],
        result[2][0], result[2][1], result[2][2]
    };

    return vec * mult;
    // glm implementarion TOSTUDY
    // {
    // T const a = angle;
    // T const c = cos(a);
    // T const s = sin(a);

    // vec<3, T, Q> axis(normalize(v));
    // vec<3, T, Q> temp((T(1) - c) * axis);
    // std::cout << temp[0] << ' ' << temp[1] << ' ' << temp[2] << ' ' << std::endl;


    // mat<4, 4, T, Q> Rotate;
    // Rotate[0][0] = c + temp[0] * axis[0];
    // Rotate[0][1] = temp[0] * axis[1] + s * axis[2];
    // Rotate[0][2] = temp[0] * axis[2] - s * axis[1];

    // Rotate[1][0] = temp[1] * axis[0] - s * axis[2];
    // Rotate[1][1] = c + temp[1] * axis[1];
    // Rotate[1][2] = temp[1] * axis[2] + s * axis[0];

    // Rotate[2][0] = temp[2] * axis[0] + s * axis[1];
    // Rotate[2][1] = temp[2] * axis[1] - s * axis[0];
    // Rotate[2][2] = c + temp[2] * axis[2];

    // mat<4, 4, T, Q> Result;
    // Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
    // Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
    // Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
    // Result[3] = m[3];

    // return Result;
    // }
}

inline float radians(float deg) {
    return deg * PI / 180;
}

inline float angle(const vec3& v1, const vec3& v2) {
    return acos(v1.dot(v2) / (v1.mag() * v2.mag()));
}

inline vec3 negate(const vec3& v) {
    return vec3(v.x * -1, v.y * -1, v.z * -1);
}

inline float mix(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

inline vec3 mix(const vec3&a, const vec3& b, float t) {
    return a * (1.0 - t) + b * t;
}

inline quat slerp(const quat& q1, const quat& q2, float t) {
    quat z;

    float omega = dot(q1, q2);

    if(omega < 0) {
        //TODO: Implement unary operators
        z = { -q2.x, -q2.y, -q2.z, -q2.w };
        omega = -omega;
    }

    //TODO: Handle double

    // GLM implementation https://github.com/g-truc/glm/blob/master/glm/ext/quaternion_common.inl
    // Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
    if(omega > 1.0f - std::numeric_limits<float>::epsilon()) {
        return {
            mix(q1.x, q2.x, t),
            mix(q1.y, q2.y, t),
            mix(q1.z, q2.z, t),
            mix(q1.w, q2.w, t)
        };
    }

    else {
        float angle = acos(omega);
        return (sin((1.0f - t) * angle) * q1 + sin(t * angle) * z) / sin(angle);
    }

}

inline float lenght(const quat& q) {
    return sqrt(dot(q, q));
}

inline quat norm(const quat& q) {
    float len = lenght(q);

    if(len <= 0.0f)
        return { 0.0f, 0.0f, 0.0f, 1.0f };

    float oneOverLen = 1.0f / len;

    return {q.x * oneOverLen, q.y * oneOverLen, q.z * oneOverLen, q.w * oneOverLen};
}

}
