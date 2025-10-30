#pragma once

/*
 * Linear Algebra Library
 * Automatic Differentiation Support
 * Author: Zoltan Simon
 * Date: October 2025
 * Description: A library for linear algebra operations with automatic differentiation support.
 * License: MIT License
 *
 */

#include <complex>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <format>

#ifdef ENABLE_CUDA_SUPPORT
    #include <cuda/std/complex>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #define CUDA_COMPATIBLE __device__ __host__
    #define CUDA_DEVICE __device__
    #define CUDA_HOST __host__
#else
    #define CUDA_COMPATIBLE
    #define CUDA_DEVICE
    #define CUDA_HOST
#endif

namespace linalg {

template <typename T>
constexpr T PI = T{3.14159265358979323846264338327950288419716939937510582097494459230781640628};


template <typename, template<typename...> class>
inline constexpr bool is_specialization_v = false;

template <template<typename...> class Template, typename... Args>
inline constexpr bool is_specialization_v<Template<Args...>, Template> = true;

#ifdef ENABLE_CUDA_SUPPORT
    template<typename T>
    concept ComplexType = is_specialization_v<T, cuda::std::complex> || is_specialization_v<T, std::complex>;
#else
    template<typename T>
    concept ComplexType = is_specialization_v<T, std::complex>;
#endif
    template<typename T>
    concept RealType = std::is_floating_point_v<T>;

    template<typename T>
    concept ScalarType = ComplexType<T> || RealType<T> || std::is_integral_v<T>;

    template<class T>
    concept DerivableType = ScalarType<T> || requires (T var) {
        var.fx();
        var.dfxdx();
    };

    template<DerivableType T>
    class Dual;

    template<typename T>
    concept DualType = is_specialization_v<T, Dual>;

    template<typename T>
    concept DualOrScalar = ScalarType<T> || DualType<T>;

    template<typename>
    struct unwrap;

    template<ScalarType T>
    struct unwrap<T> {
        using type = T;
    };

    template<DualOrScalar T>
    struct unwrap<Dual<T>> {
        using type = T;
    };

    template<typename T>
    using unwrap_t = typename unwrap<T>::type;

    // Up to derivation order two
    template<typename T>
    concept DualOfReal = DualType<T> && RealType<unwrap_t<T>> || DualType<T> && DualType<unwrap_t<T>> && RealType<unwrap_t<unwrap_t<T>>>;

    // Up to derivation order two
    template<typename T>
    concept DualOfComplex = DualType<T> && ComplexType<unwrap_t<T>> || DualType<T> && DualType<unwrap_t<T>> && ComplexType<unwrap_t<unwrap_t<T>>>;

    template<typename T>
    concept DualOfRealOrRealScalar = RealType<T> || DualOfReal<T>;

    template<typename T>
    concept DualOfComplexOrComplexScalar = ComplexType<T> || DualOfComplex<T>;

    template<typename T, typename U>
    using LargerOrDerivative
        = std::conditional_t<
            (is_specialization_v<T, Dual> || is_specialization_v<U, Dual>),
            Dual<
                std::conditional_t<
                    (is_specialization_v<unwrap_t<T>, Dual> || is_specialization_v<unwrap_t<U>, Dual>),
                    Dual<
                        std::conditional_t<
                            (sizeof(unwrap_t<unwrap_t<T>>) > sizeof(unwrap_t<unwrap_t<U>>)),
                            unwrap_t<unwrap_t<T>>,
                            unwrap_t<unwrap_t<U>>
                        >
                    >,
                    std::conditional_t<
                        (sizeof(unwrap_t<T>) > sizeof(unwrap_t<U>)),
                        unwrap_t<T>,
                        unwrap_t<U>
                    >
                >
            >,
            std::conditional_t<(sizeof(T) > sizeof(U)), T, U>
        >;   // Let the type be Dual if one of the types is derivative and select the larger precision of the two inner types.


    template<DualOrScalar T>
    using ForceComplex 
        = std::conditional_t<
            DualType<T>,
            Dual<std::conditional_t<
                    is_specialization_v<unwrap_t<T>, std::complex>,
                    unwrap_t<T>,
                    std::complex<unwrap_t<T>>
                >
            >,
            std::conditional_t<
                is_specialization_v<T, std::complex>,
                T,
                std::complex<T>
            >
        >;


    /*
    Calculates the value of a function and its derivative with respect to x.
    Represents (f(x), df/dx)
    The derivative is assumed to be with respect to the right-hand side variable.
    */
    template<DerivableType T>
    class Dual {
        public:
    
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr Dual(T _fx = T{}, T _dfxdx = static_cast<T>(0.0)): m_fx{_fx}, m_dfxdx{_dfxdx} {
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr Dual(const Dual<T>& pair): m_fx{pair.m_fx}, m_dfxdx{pair.m_dfxdx} {
        }
        
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto& operator=(const Dual<T>& pair) {
            this->m_fx = pair.m_fx;
            this->m_dfxdx = pair.m_dfxdx;
            return *this;
        }

        template<ScalarType T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr explicit operator Dual<T2>() const {
            return Dual<T2>{static_cast<T2>(m_fx), static_cast<T2>(m_dfxdx)};
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator< (const Dual<T2>& dual) const {
            return this->fx() < dual.fx();
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator> (const Dual<T2>& dual) const {
            return this->fx() > dual.fx();
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator== (const Dual<T2>& dual) const {
            return this->fx() == dual.fx();
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator+(const Dual<T2>& pair) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx + pair._raw_fx(), m_dfxdx + pair._raw_dfxdx() };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator-(const Dual<T2>& pair) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx - pair._raw_fx(), m_dfxdx - pair._raw_dfxdx() };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator*(const Dual<T2>& pair) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx * pair._raw_fx(), m_fx * pair._raw_dfxdx() + m_dfxdx * pair._raw_fx() };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator/(const Dual<T2>& pair) const {
            return Dual<LargerOrDerivative<T, T2>>{
                m_fx / pair._raw_fx(),
                (m_dfxdx * pair._raw_fx() - m_fx * pair._raw_dfxdx()) / (pair._raw_fx() * pair._raw_fx())
            };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator+(T2 a) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx + a, m_dfxdx };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator-(T2 a) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx - a, m_dfxdx };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator*(T2 a) const {
            return Dual<LargerOrDerivative<T, T2>>{ m_fx * a, m_dfxdx * a };
        }

        template<DualOrScalar T2>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator/(T2 a) const {
            return Dual<LargerOrDerivative<T, T2>>{
                m_fx / a,
                (m_dfxdx * a - m_fx * a) / (a * a)
            };
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto operator-() const {
            return Dual{ -m_fx, -m_dfxdx };
        }

        CUDA_COMPATIBLE constexpr auto& operator+=(T a) {
            this->m_fx += a;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator-=(T a) {
            this->m_fx -= a;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator*=(T a) {
            this->m_fx *= a;
            this->m_dfxdx *= a;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator/=(T a) {
            this->m_fx /= a;
            this->m_dfxdx /= a;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator+=(const Dual& pair) {
            this->m_fx += pair.m_fx;
            this->m_dfxdx += pair.m_dfxdx;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator-=(const Dual& pair) {
            this->m_fx -= pair.m_fx;
            this->m_dfxdx -= pair.m_dfxdx;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator*=(const Dual& pair) {
            this->m_fx *= pair.m_fx;
            this->m_dfxdx = this->m_fx * pair.m_dfxdx + this->m_dfxdx * pair.m_fx;
            return *this;
        }

        CUDA_COMPATIBLE constexpr auto& operator/=(const Dual& pair) {
            this->m_fx /= pair.m_fx;
            this->m_dfxdx = (this->m_dfxdx * pair.m_fx - this->m_fx * pair.m_dfxdx) / (pair.m_fx * pair.m_fx);
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto _raw_fx() const {
            return m_fx;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto _raw_dfxdx() const {
            return m_dfxdx;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto fx() const {
            if constexpr (ScalarType<T>) {
                return m_fx;
            } else {
                return m_fx.fx();
            }
        }

        /*
        Returns the derivative of order 'order'.
        */
        template<uint8_t order = 1>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto dfxdx() const {
            if constexpr (order == 0) {
                return fx();
            } else if constexpr (ScalarType<T>) {
                if constexpr (order == 1) {
                    return m_dfxdx;
                } else {
                    return static_cast<T>(0.0);
                }
            } else {
                if constexpr (order == 1) {
                    return m_dfxdx.fx();
                } else {
                    return m_dfxdx.dfxdx<order - 1>();
                }
            }
        }

        private:
        T m_fx;
        T m_dfxdx;  // Right hand side derivative
    };

    template<ScalarType T>
    using FirstOrderDerivative = Dual<T>;
    template<ScalarType T>
    using SecondOrderDerivative = Dual<Dual<T>>;
    template<ScalarType T>
    using ThirdOrderDerivative = Dual<Dual<Dual<T>>>;

    template<ScalarType T, uint32_t Order> requires (Order >= 1)
    CUDA_COMPATIBLE static constexpr auto initDerivativePair(T value) {
        if constexpr (Order == 1) {
            return FirstOrderDerivative<T>{value, 1.0};
        } else if constexpr (Order == 2) {
            return SecondOrderDerivative<T>{
                Dual<T>{value, static_cast<T>(1.0) },
                Dual<T>{static_cast<T>(1.0), static_cast<T>(0.0)} };
        }
        else if constexpr (Order == 3) {
            return ThirdOrderDerivative<T>{
                Dual<Dual<T>>{Dual<T>{value, static_cast<T>(1.0)}, Dual<T>{static_cast<T>(1.0), static_cast<T>(0.0)}},
                Dual<Dual<T>>{Dual<T>{static_cast<T>(1.0), static_cast<T>(0.0)}, Dual<T>{static_cast<T>(0.0), static_cast<T>(0.0)}}
            };
        }
        else {
            static_assert(Order <= 3, "Only supports up to third order derivatives.");
        }
    }

    template<ScalarType T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE auto operator+(T a, const Dual<U>& pair) {
        return Dual{ pair._raw_fx() + a, pair._raw_dfxdx() };
    }

    template<ScalarType T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE auto operator-(T a, const Dual<U>& pair) {
        return Dual{ pair._raw_fx() - a, pair._raw_dfxdx() };
    }

    template<ScalarType T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE auto operator*(T a, const Dual<U>& pair) {
        return Dual{ pair._raw_fx() * a, pair._raw_dfxdx() * a };
    }

    template<ScalarType T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE auto operator/(T a, const Dual<U>& pair) {
        return Dual{
            a / pair._raw_fx(),
            - pair._raw_dfxdx() / pair._raw_fx() / pair._raw_fx() * a
        };
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto exp(const Dual<T>& pair) {
        T exp_fx;
        if constexpr (ScalarType<T>) {
            exp_fx = std::exp(pair._raw_fx());
        } else {
            exp_fx = linalg::exp(pair._raw_fx());
        }
        return Dual<T>{
            exp_fx,
            exp_fx * pair._raw_dfxdx()
        };
    }

    template<DualOrScalar T, ScalarType U>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto pow(const Dual<T>& base, U exponent) {
        if (base.fx() == decltype(base.fx()){}) {
            return Dual<T>{T{}, T{}};
        }
        T pow_fx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            pow_fx = cuda::std::pow(base._raw_fx(), exponent);
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            pow_fx = std::pow(base._raw_fx(), exponent);
        }
        else {
            pow_fx = linalg::pow(base._raw_fx(), exponent);
        }
        return Dual<T>{
            pow_fx,
            pow_fx * static_cast<T>(exponent) * base._raw_dfxdx() / base._raw_fx()
        };
    }

    template<ScalarType T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto pow(T base, const Dual<U>& exponent) {
        U pow_fx;
        if constexpr (ScalarType<U>) {
            pow_fx = std::pow(base, exponent._raw_fx());
        }
        else {
            pow_fx = linalg::pow(base, exponent._raw_fx());
        }
        return Dual<U>{
            pow_fx,
            pow_fx * log(base) * exponent._raw_dfxdx()
        };
    }

    template<DualOrScalar T, DualOrScalar U>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto pow(const Dual<T>& base, const Dual<U>& exponent) {
        LargerOrDerivative<T, U> pow_fx;
        if constexpr (ScalarType<T> && ScalarType<U>) {
            pow_fx = std::pow(base._raw_fx(), exponent._raw_fx());
        } else {
            pow_fx = linalg::pow(base._raw_fx(), exponent._raw_fx());
        }
        return Dual{
            pow_fx,
            pow_fx * (base._raw_dfxdx() * exponent._raw_fx() / base._raw_fx() + exponent._raw_dfxdx() * log(base._raw_fx()))
        };
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto sqrt(const Dual<T>& pair) {
        T sqrt_fx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            sqrt_fx = cuda::std::sqrt(pair._raw_fx());
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            sqrt_fx = std::sqrt(pair._raw_fx());
        } else {
            sqrt_fx = linalg::sqrt(pair._raw_fx());
        }
        return Dual{
            sqrt_fx,
            pair._raw_dfxdx() / (static_cast<T>(2.0) * sqrt_fx)
        };
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto log(const Dual<T>& exponent) {
        T log_fx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            log_fx = cuda::std::log(exponent._raw_fx());
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            log_fx = std::log(exponent._raw_fx());
        } else {
            log_fx = linalg::log(exponent._raw_fx());
        }
        return Dual{
            log_fx,
            exponent._raw_dfxdx() / exponent._raw_fx()
        };
    }


    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto cos(const Dual<T>& pair) noexcept;

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto sin(const Dual<T>& pair) noexcept {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::sin(pair._raw_fx()),
                cuda::std::cos(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::sin(pair._raw_fx()),
                std::cos(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else {
            return Dual{
                linalg::sin(pair._raw_fx()),
                linalg::cos(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto cos(const Dual<T>& pair) noexcept {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::cos(pair._raw_fx()),
                -cuda::std::sin(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::cos(pair._raw_fx()),
                -std::sin(pair._raw_fx()) * pair._raw_dfxdx()
            };
        } else {
            return Dual{
                linalg::cos(pair._raw_fx()),
                -linalg::sin(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto tan(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::tan(pair._raw_fx()),
                pair._raw_dfxdx() / (cuda::std::cos(pair._raw_fx()) * cuda::std::cos(pair._raw_fx()))
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::tan(pair._raw_fx()),
                pair._raw_dfxdx() / (std::cos(pair._raw_fx()) * std::cos(pair._raw_fx()))
            };
        }
        else {
            return Dual{
                linalg::tan(pair._raw_fx()),
                pair._raw_dfxdx() / (linalg::cos(pair._raw_fx()) * linalg::cos(pair._raw_fx()))
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto asin(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::asin(pair._raw_fx()),
                pair._raw_dfxdx() / cuda::std::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::asin(pair._raw_fx()),
                pair._raw_dfxdx() / std::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        } else {
            return Dual{
                linalg::asin(pair._raw_fx()),
                pair._raw_dfxdx() / linalg::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto acos(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::acos(pair._raw_fx()),
                -pair._raw_dfxdx() / cuda::std::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::acos(pair._raw_fx()),
                -pair._raw_dfxdx() / std::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        } else {
            return Dual{
                linalg::acos(pair._raw_fx()),
                -pair._raw_dfxdx() / linalg::sqrt(T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto atan(const Dual<T>& pair) {
        T atan_fx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            atan_fx = cuda::std::atan(pair._raw_fx());
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            atan_fx = std::atan(pair._raw_fx());
        }
        else {
            atan_fx = linalg::atan(pair._raw_fx());
        }
        return Dual{
            atan_fx,
            pair._raw_dfxdx() / (T{1.0} + pair._raw_fx() * pair._raw_fx())
        };
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto cosh(const Dual<T>& pair);

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto sinh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::sinh(pair._raw_fx()),
                cuda::std::cosh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::sinh(pair._raw_fx()),
                std::cosh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else {
            return Dual{
                linalg::sinh(pair._raw_fx()),
                linalg::cosh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto cosh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::cosh(pair._raw_fx()),
                cuda::std::sinh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::cosh(pair._raw_fx()),
                std::sinh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
        else {
            return Dual<T>{
                linalg::cosh(pair._raw_fx()),
                linalg::sinh(pair._raw_fx()) * pair._raw_dfxdx()
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto tanh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::tanh(pair._raw_fx()),
                pair._raw_dfxdx() * (T{1.0} - cuda::std::tanh(pair._raw_fx()) * cuda::std::tanh(pair._raw_fx()))
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::tanh(pair._raw_fx()),
                pair._raw_dfxdx() * (T{1.0} - std::tanh(pair._raw_fx()) * std::tanh(pair._raw_fx()))
            };
        }
        else {
            return Dual{
                linalg::tanh(pair._raw_fx()),
                pair._raw_dfxdx() * (T{1.0} - linalg::tanh(pair._raw_fx()) * linalg::tanh(pair._raw_fx()))
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto asinh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::asinh(pair._raw_fx()),
                pair._raw_dfxdx() / cuda::std::sqrt(T{1.0} + pair._raw_fx() * pair._raw_fx())
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::asinh(pair._raw_fx()),
                pair._raw_dfxdx() / std::sqrt(T{1.0} + pair._raw_fx() * pair._raw_fx())
            };
        } else {
            return Dual{
                linalg::asinh(pair._raw_fx()),
                pair._raw_dfxdx() / linalg::sqrt(T{1.0} + pair._raw_fx() * pair._raw_fx())
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto acosh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::acosh(pair._raw_fx()),
                pair._raw_dfxdx() / cuda::std::sqrt(pair._raw_fx() * pair._raw_fx() - T{1.0})
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::acosh(pair._raw_fx()),
                pair._raw_dfxdx() / std::sqrt(pair._raw_fx() * pair._raw_fx() - T{1.0})
            };
        } else {
            return Dual{
                linalg::acosh(pair._raw_fx()),
                pair._raw_dfxdx() / linalg::sqrt(pair._raw_fx() * pair._raw_fx() - T{1.0})
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto atanh(const Dual<T>& pair) {
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            return Dual{
                cuda::std::atanh(pair._raw_fx()),
                pair._raw_dfxdx() / (T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            return Dual{
                std::atanh(pair._raw_fx()),
                pair._raw_dfxdx() / (T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        } else {
            return Dual{
                linalg::atanh(pair._raw_fx()),
                pair._raw_dfxdx() / (T{1.0} - pair._raw_fx() * pair._raw_fx())
            };
        }
    }

    template<DualOrScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto abs(const Dual<T>& pair) {
        T abs_fx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            abs_fx = cuda::std::abs(pair._raw_fx());
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            abs_fx = std::abs(pair._raw_fx());
        } else {
            abs_fx = linalg::abs(pair._raw_fx());
        }
        return Dual<T>{
            abs_fx,
            abs_fx / pair._raw_fx() * pair._raw_dfxdx()
        };
    }

    template<DualOfRealOrRealScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr inline auto conj(const Dual<T>& pair) {
        return pair;
    }

    template<DualOfComplexOrComplexScalar T>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr inline auto conj(const Dual<T>& pair) {
        T conj_fx, conj_dfxdx;
        #ifdef ENABLE_CUDA_SUPPORT
        if constexpr (is_specialization_v<T, cuda::std::complex>) {
            conj_fx = cuda::std::conj(pair._raw_fx());
            conj_dfxdx = cuda::std::conj(pair._raw_dfxdx());
        }
        else
        #endif
        if constexpr (ScalarType<T>) {
            conj_fx = std::conj(pair._raw_fx());
            conj_dfxdx = std::conj(pair._raw_dfxdx());
        } else {
            conj_fx = linalg::conj(pair._raw_fx());
            conj_dfxdx = linalg::conj(pair._raw_dfxdx());
        }
        return Dual<T>{
            conj_fx,
            conj_dfxdx
        };
    }

    template<DualOrScalar T, uint32_t N> requires( 0 != N )
    class Vec {
    public:
    
        CUDA_COMPATIBLE
        [[nodiscard]]
        static constexpr Vec filled(T value) {
            Vec result;
            for (uint32_t i{0}; i < N; ++i) {
                result[i] = value;
            }
            return result;
        }

        CUDA_COMPATIBLE
        [[nodiscard]]
        constexpr Vec() : m_data{} {}


        // initializer-list constructor: allow construction from brace lists
        // e.g. Vec<double,2>{-5.0, 0}
        template<DualOrScalar U>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr explicit Vec(std::initializer_list<U> list) noexcept {
            uint32_t i = 0;
            for (auto it = list.begin(); it != list.end() && i < N; ++it, ++i) {
                m_data[i] = static_cast<T>(*it);
            }
            for (; i < N; ++i) {
                m_data[i] = T{};
            }
        }


        template<DualOrScalar U>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr explicit Vec(const Vec<U, N>& v) noexcept {
            for (int i = 0; i < N; i++) {
                m_data[i] = static_cast<T>(v[i]);
            }
        }

        CUDA_COMPATIBLE
        [[nodiscard]]
        constexpr inline uint32_t dim() const noexcept {
            return N;
        }

        CUDA_COMPATIBLE
        [[nodiscard]]
        constexpr const T* data() const noexcept {
            return this->m_data;
        }

        CUDA_COMPATIBLE
        [[nodiscard]]
        constexpr T* data() noexcept {
            return this->m_data;
        }

        CUDA_COMPATIBLE constexpr Vec& operator=(const Vec& v) {
            for (int i = 0; i < N; i++) {
                m_data[i] = v[i];
            }
            return *this;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator+(const Vec<U, N>& v) const {
            Vec<decltype(this->m_data[0] + v[0]), N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] + v[i];
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator-(const Vec<U, N>& v) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] - v[i];
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator/(const Vec<U, N>& v) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] / v[i];
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator+(U scalar) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] + scalar;
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator-(U scalar) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] - scalar;
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator*(U scalar) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] * scalar;
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr auto operator/(U scalar) const {
            Vec<LargerOrDerivative<T, U>, N> w;
            for (int i = 0; i < N; i++) {
                w[i] = this->m_data[i] / scalar;
            }
            return w;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr Vec& operator+=(const Vec<U, N>& v) {
            for (int i = 0; i < N; i++) {
                m_data[i] += v[i];
            }
            return *this;
        }

        template<DualOrScalar U>
        CUDA_COMPATIBLE constexpr Vec& operator-=(const Vec<U, N>& v) {
            for (int i = 0; i < N; i++) {
                m_data[i] -= v[i];
            }
            return *this;
        }

        CUDA_COMPATIBLE constexpr inline T& operator[](unsigned int idx) {
            return m_data[idx];
        }

        CUDA_COMPATIBLE constexpr inline const T& operator[](unsigned int idx) const {
            return m_data[idx];
        }

        CUDA_COMPATIBLE constexpr inline T& x() {
            return m_data[0]; 
        }

        CUDA_COMPATIBLE constexpr inline const T& x() const {
            return m_data[0]; 
        }

        CUDA_COMPATIBLE constexpr inline T& y() {
            return m_data[1]; 
        }

        CUDA_COMPATIBLE constexpr inline const T& y() const {
            return m_data[1]; 
        }

        CUDA_COMPATIBLE constexpr inline T& z() {
            if constexpr(N < 3) {
                static_assert("This vector has no z coordinate!");
            }
            return m_data[2]; 
        }

        CUDA_COMPATIBLE constexpr inline const T& z() const {
            if constexpr(N < 3) {
                static_assert("This vector has no z coordinate!");
            }
            return m_data[2]; 
        }

        CUDA_COMPATIBLE constexpr inline T& w() {
            if constexpr(N < 4) {
                static_assert("This vector has no w coordinate!");
            }
            return m_data[3]; 
        }

        CUDA_COMPATIBLE constexpr inline const T& w() const {
            if constexpr(N < 4) {
                static_assert("This vector has no w coordinate!");
            }
            return m_data[3]; 
        }

        CUDA_COMPATIBLE constexpr inline T& a() {
            if constexpr(N < 4) {
                static_assert("This vector has no alpha coordinate!");
            }
            return m_data[3]; 
        }

        CUDA_COMPATIBLE constexpr inline const T& a() const {
            if constexpr(N < 4) {
                static_assert("This vector has no alpha coordinate!");
            }
            return m_data[3]; 
        }

        CUDA_HOST operator std::string() const
        {
            std::string str = "(";
            for (int i = 0; i < N; i++) {
                str.append(std::to_string(m_data[i]));
                if (i < N - 1)
                    str.append(", ");
            }
            str.append(")");
            return str;
        }

        CUDA_HOST operator std::string_view() const
        {
            std::string str = "(";
            for (int i = 0; i < N; i++) {
                str.append(std::to_string(m_data[i]));
                if (i < N - 1)
                    str.append(", ");
            }
            str.append(")");
            return str;
        }

        CUDA_COMPATIBLE operator const char*() const {
            return this->operator std::string().c_str();
        }

        template<typename T>
        CUDA_COMPATIBLE constexpr operator const Vec<T, N>() const {
            Vec<T, N> v;
            for (uint32_t i = 0; i < N; ++i) {
                v[i] = (T)m_data[i];
            }
            return v;
        }

        template<uint32_t _N>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr Vec<T, _N> cropDimension() const {
            static_assert(_N <= N, "Cannot crop to a larger dimension!");
            Vec<T, _N> v;
            for (uint32_t i = 0; i < _N; ++i) {
                v[i] = m_data[i];
            }
            return v;
        }

    protected:
        T m_data[N];
    };
    
    template<DualOrScalar T, uint32_t N>
    CUDA_HOST std::ostream& operator<<(std::ostream& stream, const Vec<T, N>& v) {
        stream << "(";
        for (int i = 0; i < N; i++) {
            stream << v[i];
            if (i < N - 1)
                stream << ", ";
        }
        stream << ")";
        return stream;
    }

        // Alias
    template<DualOrScalar T>
    using Vec2 = Vec<T, 2>;

    template<DualOrScalar T>
    using Vec3 = Vec<T, 3>;

    template<DualOrScalar T>
    using Vec4 = Vec<T, 4>;

    using UVec2 = Vec2<uint32_t>;

    using UVec3 = Vec3<uint32_t>;

    using UVec4 = Vec4<uint32_t>;
    
    using IVec2 = Vec2<int32_t>;

    using IVec3 = Vec3<int32_t>;

    using IVec4 = Vec4<int32_t>;

    /*
        Dot product of two real valued vectors 
    */
    template<DualOrScalar T, DualOfRealOrRealScalar U,  uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto dot(const Vec<T, N>& u, const Vec<U, N>& v) {
        auto sum = LargerOrDerivative<T, U>{};
        for (int i = 0; i < N; i++) {
            sum += u[i] * v[i];
        }
        return sum;
    }

    /*
        Dot product of two complex valued vectors
    */
    template<DualOrScalar T, DualOfComplexOrComplexScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto dot(const Vec<T, N>& u, const Vec<U, N>& v) {
        auto sum = LargerOrDerivative<T, U>{};
        for (int i = 0; i < N; i++) {
            sum += u[i] * conj(v[i]);
        }
        return sum;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto operator*(const Vec<T, N>& u, const Vec<U, N>& v) {
        return dot(u, v);
    }

    
    // Dot product of two real valued vectors 
    template<DualOrScalar T, DualOfRealOrRealScalar U,  uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto dot(const Mat<T, 1, N>& u, const Mat<U, N, 1>& v) {
        auto sum = LargerOrDerivative<T, U>{};
        for (int i = 0; i < N; i++) {
            sum += u[0][i] * v[i][0];
        }
        return sum;
    }

    /*
    // Dot product of two complex valued vectors
    template<DualOrScalar T, DualOfComplexOrComplexScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto dot(const Mat<T, 1, N>& u, const Mat<U, N, 1>& v) {
        auto sum = LargerOrDerivative<T, U>{};
        for (int i = 0; i < N; i++) {
            sum += u[0][i] * conj(v[i][0]);
        }
        return sum;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr inline auto operator*(const Mat<T, 1, N>& u, const Mat<U, N, 1>& v) {
        return dot(u, v);
    }
    */
    
    template<DualOrScalar T, DualOrScalar U>
    CUDA_COMPATIBLE constexpr auto cross(const Vec<T, 3>& u, const Vec<U, 3>& v) {
        return Vec<LargerOrDerivative<T, U>, 3>{
            u.y() * v.z() - u.z() * v.y(),
            u.z() * v.x() - u.x() * v.z(),
            u.x() * v.y() - u.y() * v.x()
        };
    }
    
    /*
        P-norm of the vector
    */
    template<uint32_t P, DualOrScalar T, uint32_t N> requires(P > 2)
    CUDA_COMPATIBLE constexpr auto norm(const Vec<T, N>& v) {
        auto sum = T{};
        for (int i = 0; i < N; i++) {
            sum += pow(abs(v[i]), static_cast<int>(P));
        }
        return pow(sum, 1.0 / static_cast<double>(P));
    }

    /*
        2-norm of the vector
    */
    template<uint32_t P, DualOrScalar T, uint32_t N> requires(P == 2)
    CUDA_COMPATIBLE constexpr auto norm(const Vec<T, N>& v) {
        auto sum = T{};
        for (int i = 0; i < N; i++) {
            sum += pow(abs(v[i]), 2);
        }
        return sqrt(sum);
    }

    /*
        1-norm of the vector
    */
    template<uint32_t P, DualOrScalar T, uint32_t N> requires(P == 1)
    CUDA_COMPATIBLE constexpr auto norm(const Vec<T, N>& v) {
        auto sum = T{};
        for (int i = 0; i < N; i++) {
            sum += abs(v[i]);
        }
        return sum;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr auto operator*(T scalar, const Vec<U, N>& v) {
        Vec<LargerOrDerivative<T, U>, N> w;
        for (int i = 0; i < N; i++) {
            w[i] = v[i] * scalar;
        }
        return w;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr auto operator+(T scalar, const Vec<U, N>& v) {
        Vec<LargerOrDerivative<T, U>, N> w;
        for (int i = 0; i < N; i++) {
            w[i] = scalar + v[i];
        }
        return w;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N>
    CUDA_COMPATIBLE constexpr auto operator-(T scalar, const Vec<U, N>& v) {
        Vec<LargerOrDerivative<T, U>, N> w;
        for (int i = 0; i < N; i++) {
            w[i] = scalar - v[i];
        }
        return w;
    }

    template<DualOrScalar T, uint32_t R, uint32_t C>  requires( 0 != R && 0 != C)
    class Mat {
        public:
            
            CUDA_COMPATIBLE
            [[nodiscard]]
            static constexpr auto identity() noexcept {
                static_assert(R == C, "Identity matrix must be square!");
                auto M = Mat();
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        M[i][j] = (i == j) ? T{1} : T{0};
                    }
                }
                return M;
            }
        
            CUDA_COMPATIBLE
            [[nodiscard]]
            constexpr Mat() noexcept : m_data{} {}

            CUDA_COMPATIBLE
            [[nodiscard]]
            constexpr Mat(std::initializer_list<Vec<T, C>> list) noexcept : m_data{list} {}

            CUDA_COMPATIBLE
            [[nodiscard]]
            constexpr Mat(std::initializer_list<std::initializer_list<T>> list) noexcept {
                uint32_t i = 0;
                for (auto it = list.begin(); it != list.end() && i < R; ++it, ++i) {
                    m_data[i] = Vec<T, C>{*it};
                }
            }

            CUDA_COMPATIBLE
            [[nodiscard]]
            constexpr Mat(std::initializer_list<T> list) noexcept : m_data{} {
                uint32_t i = 0;
                for (auto it = list.begin(); it != list.end() && i < R; ++it, ++i) {
                    uint32_t j = 0;
                    for (auto jt = it->begin(); jt != it->end() && j < C; ++jt, ++j) {
                        m_data[i][j] = *jt;
                    }
                    for (; j < C; ++j) {
                        m_data[i][j] = T{};
                    }
                }
                for (; i < R; ++i) {
                    for (uint32_t j = 0; j < C; ++j) {
                        m_data[i][j] = T{};
                    }
                }
            }

            CUDA_COMPATIBLE [[nodiscard]] constexpr inline uint32_t rowCount() const noexcept {
                return R;
            }

            CUDA_COMPATIBLE [[nodiscard]] constexpr inline uint32_t colCount() const noexcept {
                return C;
            }

            CUDA_COMPATIBLE [[nodiscard]] constexpr inline const Vec<T, C>& operator[](unsigned int idx) const noexcept {
                return m_data[idx];
            }

            CUDA_COMPATIBLE [[nodiscard]] constexpr inline Vec<T, C>& operator[](unsigned int idx) noexcept {
                return m_data[idx];
            }

            template<DualOrScalar U, uint32_t R2, uint32_t C2>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator*(const Mat<U, R2, C2>& other) const noexcept {
                static_assert(C == R2, "Matrix dimensions do not match for multiplication!");
                Mat<LargerOrDerivative<T, U>, R, C2> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = T{};
                        for (uint32_t k{0}; k < C; ++k) {
                            result[i][j] += m_data[i][k] * other[k][j];
                        }
                    }
                }
                return result;
            }

            template<DualOrScalar U, uint32_t N>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator*(const Vec<U, N>& vec) const {
                static_assert(C == N, "Matrix and vector dimensions do not match for multiplication!");
                Vec<LargerOrDerivative<T, U>, R> result;
                for (uint32_t i{0}; i < R; ++i) {
                    result[i] = T{};
                    for (uint32_t k{0}; k < C; ++k) {
                        result[i] += m_data[i][k] * vec[k];
                    }
                }
                return result;
            }

            template<DualOrScalar U>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator*(U scalar) const {
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] * scalar;
                    }
                }
                return result;
            }

            template<DualOrScalar U>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator/(U scalar) const {
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] / scalar;
                    }
                }
                return result;
            }
            
            template<DualOrScalar U>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator+(U scalar) const {
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] + scalar;
                    }
                }
                return result;
            }

            template<DualOrScalar U>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator-(U scalar) const {
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] - scalar;
                    }
                }
                return result;
            }

            template<DualOrScalar U, uint32_t R2, uint32_t C2>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator+(const Mat<U, R2, C2>& other) const {
                static_assert(R == R2 && C == C2, "Matrix dimensions do not match for addition!");
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] + other[i][j];
                    }
                }
                return result;
            }

            template<DualOrScalar U, uint32_t R2, uint32_t C2>
            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator-(const Mat<U, R2, C2>& other) const {
                static_assert(R == R2 && C == C2, "Matrix dimensions do not match for subtraction!");
                Mat<LargerOrDerivative<T, U>, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = m_data[i][j] - other[i][j];
                    }
                }
                return result;
            }

            CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto operator-() const {
                Mat<T, R, C> result;
                for (uint32_t i{0}; i < R; ++i) {
                    for (uint32_t j{0}; j < C; ++j) {
                        result[i][j] = -m_data[i][j];
                    }
                }
                return result;
            }

            private:
            Vec<T, C> m_data[R];
    };

    template<DualOrScalar T, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto transpose(const Mat<T, R, C>& mat) {
        Mat<T, C, R> result;
        for (uint32_t i{0}; i < R; ++i) {
            for (uint32_t j{0}; j < C; ++j) {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    template<DualOrScalar T, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto conj(const Mat<T, R, C>& mat) {
        Mat<T, R, C> result;
        for (uint32_t i{0}; i < R; ++i) {
            for (uint32_t j{0}; j < C; ++j) {
                if constexpr (is_specialization_v<T, std::complex>) {
                    result[j][i] = std::conj(mat[i][j]);
                } else {
                    result[j][i] = mat[i][j];
                }
            }
        }
        return result;
    }

    template<DualOrScalar T, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto adj(const Mat<T, R, C>& mat) {
        return transpose(conj(mat));
    }

    template<DualOrScalar T, uint32_t N>
    CUDA_COMPATIBLE [[nodiscard]] constexpr inline auto adj(const Vec<T, N>& vec) {
        return conj(vec);
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t N, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr auto operator*(const Vec<T, N>& vec, const Mat<U, R, C>& mat) noexcept {
        static_assert(N == R, "Matrix and vector dimensions do not match for multiplication!");
        Vec<LargerOrDerivative<T, U>, C> result;
        for (uint32_t i{0}; i < C; ++i) {
            result[i] = T{};
            for (uint32_t k{0}; k < R; ++k) {
                result[i] += vec[k] * mat[k][i];
            }
        }
        return result;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t R1, uint32_t C1, uint32_t R2, uint32_t C2>
    CUDA_COMPATIBLE [[nodiscard]] constexpr auto kronecker(const Mat<T, R1, C1>& A, const Mat<U, R2, C2>& B) noexcept {
        Mat<LargerOrDerivative<T, U>, R1 * R2, C1 * C2> result;
        for (uint32_t i{0}; i < R2; ++i) {
            for (uint32_t j{0}; j < C2; ++j) {
                for (uint32_t k{0}; k < R1; ++k) {
                    for (uint32_t l{0}; l < C1; ++l) {
                        result[i * R2 + k][j * C2 + l] = A[k][l] * B[i][j];
                    }
                }
            }
        }
        return result;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr auto operator*(T scalar, const Mat<U, R, C>& mat) {
        Mat<LargerOrDerivative<T, U>, R, C> result;
        for (uint32_t i{0}; i < R; ++i) {
            for (uint32_t j{0}; j < C; ++j) {
                result[i][j] = scalar * mat[i][j];
            }
        }
        return result;
    }
    
    template<DualOrScalar T, DualOrScalar U, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr auto operator+(T scalar, const Mat<U, R, C>& mat) {
        Mat<LargerOrDerivative<T, U>, R, C> result;
        for (uint32_t i{0}; i < R; ++i) {
            for (uint32_t j{0}; j < C; ++j) {
                result[i][j] = scalar + mat[i][j];
            }
        }
        return result;
    }

    template<DualOrScalar T, DualOrScalar U, uint32_t R, uint32_t C>
    CUDA_COMPATIBLE [[nodiscard]] constexpr auto operator-(T scalar, const Mat<U, R, C>& mat) {
        Mat<LargerOrDerivative<T, U>, R, C> result;
        for (uint32_t i{0}; i < R; ++i) {
            for (uint32_t j{0}; j < C; ++j) {
                result[i][j] = scalar - mat[i][j];
            }
        }
        return result;
    }

    template<DualOrScalar ...T>
    struct EquationSolution {
        std::tuple<T...> roots;
        uint16_t root_count;
    };

    enum class RootDomain {
        Real,
        Complex
    };

    /*
    Return the number of roots found while solving the quadratic equation ax^2 + bx + c = 0:
    As a sideeffect, the roots are written to root_min and root_plus
    2: two real roots
    1: one real root
    0: no real roots
    */
    template<RootDomain Domain = RootDomain::Real, DualOrScalar T1, DualOrScalar T2, DualOrScalar T3>
    CUDA_COMPATIBLE
    constexpr auto solveQuadraticEquation(const T1& a, const T2& b, const T3& c) noexcept {
        using outType = std::conditional_t<
                (Domain == RootDomain::Complex),
                ForceComplex<LargerOrDerivative<T1, LargerOrDerivative<T2, T3>>>,
                LargerOrDerivative<T1, LargerOrDerivative<T2, T3>>
            >;
        EquationSolution<outType, outType> solution;

        auto discriminant = b * b - 4.0 * a * c;
        if constexpr (Domain == RootDomain::Real) {
            if constexpr (ComplexType<decltype(discriminant)>) {
                if (discriminant.real() < 0.0) {
                    solution.roots = std::make_tuple(outType{}, outType{});
                    solution.root_count = 0;
                    return solution;
                }
            }
            else {
                if (discriminant < 0.0) {
                    solution.roots = std::make_tuple(outType{}, outType{});
                    solution.root_count = 0;
                    return solution;
                }
            }
        }
        if (discriminant == outType{0.0}) {
            solution.roots = std::make_tuple(-b / (static_cast<outType>(2.0) * a), -b / (static_cast<outType>(2.0) * a));
            solution.root_count = 1;
            return solution;
        }
        auto sqrt_discriminant = std::conditional_t<(Domain == RootDomain::Complex), ForceComplex<decltype(discriminant)>, decltype(discriminant)>{};
        if constexpr (ScalarType<decltype(discriminant)>) {
            if constexpr (Domain == RootDomain::Complex) {
                sqrt_discriminant = std::sqrt(ForceComplex<decltype(discriminant)>{discriminant});
            }
            else {
                sqrt_discriminant = std::sqrt(discriminant);
            }
        }
        else {
            sqrt_discriminant = linalg::sqrt(discriminant);
        }
        solution.roots = std::make_tuple((-b - sqrt_discriminant) / (static_cast<T1>(2.0) * a), (-b + sqrt_discriminant) / (static_cast<T1>(2.0) * a));
        solution.root_count = 2;
        return solution;
    }

    template<DualOrScalar T>
    auto solveCubicEquation(const T& a, const T& b, const T& c, const T& d, T& root1, T& root2, T& root3) {
        // Use Cardano's method or other numerical methods to find the roots
        static_assert("Unimplemented function!");
        
        return 0;
    }

    template<RealType T>
    CUDA_HOST std::string to_string(const T& val) {
        return std::format("{}", val);
    }

    template<ComplexType T>
    CUDA_HOST std::string to_string(const T& cVal) {
        return std::format("{}{:+}i", cVal.real(), cVal.imag());
    }

    template<DualType T>
    CUDA_HOST std::string to_string(const T& dVal) {
        if constexpr (is_specialization_v<unwrap_t<T>, Dual>) {
            return std::format("(fx={}, fx'={}, fx''={})", to_string(dVal.fx()), to_string(dVal.dfxdx()), to_string(dVal.dfxdx<2>()));
        }
        return std::format("(fx={}, fx'={})", to_string(dVal.fx()), to_string(dVal.dfxdx()));
    }

    template<DualOrScalar T, uint32_t N>
    CUDA_HOST std::string to_string(const Vec<T, N>& v) {
        std::string str = "(";
        for (int i = 0; i < N; i++) {
            str.append(to_string(v[i]));
            if (i < N - 1)
                str.append(", ");
        }
        str.append(")");
        return str;
    }

    template<DualOrScalar T, uint32_t R, uint32_t C>
    CUDA_HOST std::string to_string(const Mat<T, R, C>& mat) {
        std::string str = "";
        for (uint32_t i{0}; i < R; ++i) {
            str.append("| ");
            for (uint32_t j{0}; j < C; ++j) {
                str.append(to_string(mat[i][j]));
                if (j < C - 1)
                    str.append(", ");
            }
            str.append(" |\n");
        }
        return str;
    }

}