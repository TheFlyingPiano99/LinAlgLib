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
#include <array>

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

template <typename T>
struct std::formatter<std::complex<T>> : std::formatter<T> {
    template <typename FormatContext>
    auto format(const std::complex<T>& c, FormatContext& ctx) const {
        auto out = ctx.out();
        out = std::formatter<T>::format(c.real(), ctx);
        out = std::format_to(out, " + ");
        out = std::formatter<T>::format(std::abs(c.imag()), ctx);
        return std::format_to(out, "i");
    }
};


namespace TinyLA {

    template <class T>
    constexpr T PI = T{3.14159265358979323846264338327950288419716939937510582097494459230781640628};


    template <class, template<typename...> class>
    inline constexpr bool is_specialization_v = false;

    template <template<class...> class Template, class... Args>
    inline constexpr bool is_specialization_v<Template<Args...>, Template> = true;

    #ifdef ENABLE_CUDA_SUPPORT
        template<class T>
        concept ComplexType = is_specialization_v<T, cuda::std::complex> || is_specialization_v<T, std::complex>;
    #else
        template<class T>
        concept ComplexType = is_specialization_v<T, std::complex>;
    #endif
    template<class T>
    concept RealType = std::is_floating_point_v<T>;

    template<class T>
    concept ScalarType = ComplexType<T> || RealType<T> || std::is_integral_v<T>;

    template <class First, class... Rest>
    CUDA_COMPATIBLE inline constexpr auto first(First firstArg, Rest...) {
        return firstArg;
    }

    using VarIDType = int16_t;

    template <VarIDType... Values>
    struct DependencyArray {
        static constexpr std::array<VarIDType, sizeof...(Values)> data{Values...};
    };

    template <VarIDType... Values>
    consteval bool contains(const DependencyArray<Values...>& arr, const VarIDType& value) {
        for (std::size_t i{}; i < sizeof...(Values); i++)
            if (arr.data[i] == value)
                return true;
        return false;
    }
    
    template <VarIDType... A1, VarIDType... A2>
    consteval auto set_union(DependencyArray<A1...>, DependencyArray<A2...>) {
        constexpr auto temp = [] {
            std::array<> result{A1...}; // start with A1
            auto size = sizeof...(A1);
            for (auto v : {A2...}) {
                bool found = false;
                for (auto existing : {A1...}) {
                    if (v == existing) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    result[size++] = v;
                }
            }
            return result;
        }();

        return []<std::size_t... Is>(std::index_sequence<Is...>) constexpr {
            return DependencyArray<temp[Is]...>{};
        }(std::make_index_sequence<sizeof...(A1) + sizeof...(A2)>{});
    }

    template <VarIDType... A1, VarIDType... A2>
    consteval auto concatenate(DependencyArray<A1...>, DependencyArray<A2...>) {
        auto array = DependencyArray<A1..., A2...>{};
        return array;
    }

    template<class AE>
    class AbstractExpr {
        public:

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            return static_cast<const AE&>(*this).derivate<varId>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return static_cast<const AE&>(*this).to_string();
        }

        static constexpr bool variable_data = false;
    };


    template<class SE>
    class AbstractScalarExpr : public AbstractExpr<AbstractScalarExpr<SE>> {
        public:

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            return static_cast<const SE&>(*this).derivate<varId>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return static_cast<const SE&>(*this).to_string();
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            return (static_cast<const SE&>(*this)).eval();
        }

    };

    template<class E>
    concept ScalarExprType = requires(const E& e) { e.to_string(); e.eval(); };

    template<ScalarType T>
    class ScalarZero : public AbstractScalarExpr<ScalarZero<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarZero() {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            if constexpr (varId == 0) {
                return ScalarZero{};
            }
            else {
                return std::tuple{ ScalarZero{}... };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return std::string("0");
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            return T{};
        }
    };

    template<ScalarType T>
    class ScalarConstant : public AbstractScalarExpr<ScalarConstant<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarConstant(T value) : m_value(value) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return ScalarZero<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return std::format("{}", m_value);
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            return m_value;
        }

        private:
        T m_value;
    };

    template<ScalarType T>
    class ScalarUnit : public AbstractScalarExpr<ScalarUnit<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarUnit() {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return ScalarZero{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return std::string("1");
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            return static_cast<T>(1);
        }
    };

    template<ScalarType T, VarIDType varId = -1>
    class Scalar : public AbstractScalarExpr<Scalar<T, varId>> {
        public:

        static constexpr bool variable_data = true;

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Scalar(const AbstractScalarExpr<_SE>& expr) : m_value(expr.eval()) {}

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Scalar(T value = T{}) : m_value{value} {}

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractScalarExpr<_SE>& expr) {
            m_value = expr.eval();
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(T value) {
            m_value = value;
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE constexpr operator T() const {
            return m_value;
        }

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            if constexpr (diffVarId == varId) {
                return ScalarUnit<T>();
            }
            else {
                return ScalarZero<T>();
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            return (varId == -1)? std::format("{}", m_value) : std::format("s_{}", varId);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            return m_value;
        }

        private:
        T m_value;
    };

    template<class E1, class E2>
    class ScalarSum : public AbstractScalarExpr<ScalarSum<E1, E2>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarSum(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero> && is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarZero<int>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero>) {
                return expr2_derivative;
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return expr1_derivative;
            }
            else {
                return ScalarSum<
                    decltype(expr1_derivative),
                    decltype(expr2_derivative)
                >{
                    expr1_derivative,
                    expr2_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            if constexpr (is_specialization_v<E1, ScalarZero> && is_specialization_v<E2, ScalarZero>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, ScalarZero>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_specialization_v<E2, ScalarZero>) {
                return m_expr1.to_string();
            }
            else {
                auto str1 = std::string(m_expr1.to_string());
                auto str2 = std::string(m_expr2.to_string());
                if (!str1.empty() && !str2.empty()) {
                    return std::format("{} + {}", str1, str2);
                }
                else if (!str1.empty()) {
                    return str1;
                }
                else {
                    return str2;
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            if constexpr (is_specialization_v<E1, ScalarZero> && is_specialization_v<E2, ScalarZero>) {
                return 0;
            }
            else if constexpr (is_specialization_v<E1, ScalarZero>) {
                return m_expr2.eval();
            }
            else if constexpr (is_specialization_v<E2, ScalarZero>) {
                return m_expr1.eval();
            }
            else {
                return m_expr1.eval() + m_expr2.eval();
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ScalarExprType E1, ScalarExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(const E1& expr1, const E2& expr2) {
        return ScalarSum<E1, E2>{expr1, expr2};
    }

    template<ScalarExprType E, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(const E& expr, S a) {
        return ScalarSum<E, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ScalarExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(S a, const E& expr) {
        return ScalarSum<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E>
    class ScalarNegate : public AbstractScalarExpr<ScalarNegate<E>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarNegate(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr_derivative), ScalarZero>) {
                return ScalarZero<int>{};
            }
            else {
                return ScalarNegate<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("-{}", str_expr);
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            if constexpr (is_specialization_v<E, ScalarZero>) {
                return 0;
            }
            else {
                return -m_expr.eval();
            }
        }

        private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
    };

    template<ScalarExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E& expr) {
        return ScalarNegate<E>{expr};
    }

    template<class E1, class E2>
    class ScalarSubtraction : public AbstractScalarExpr<ScalarSubtraction<E1, E2>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarSubtraction(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero> && is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarZero<int>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero>) {
                return ScalarNegate<decltype(expr2_derivative)>{expr2_derivative};
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return expr1_derivative;
            }
            else {
                return ScalarSubtraction<
                    decltype(expr1_derivative),
                    decltype(expr2_derivative)
                >{
                    expr1_derivative,
                    expr2_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            if constexpr (is_specialization_v<E1, ScalarZero> && is_specialization_v<E2, ScalarZero>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, ScalarZero>) {
                return std::format("- ", m_expr2.to_string());
            }
            else if constexpr (is_specialization_v<E2, ScalarZero>) {
                return m_expr1.to_string();
            }
            else {
                auto str1 = std::string(m_expr1.to_string());
                auto str2 = std::string(m_expr2.to_string());
                if (!str1.empty() && !str2.empty()) {
                    return std::format("{} - {}", str1, str2);
                }
                else if (!str1.empty()) {
                    return str1;
                }
                else {
                    return str2;
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            if constexpr (is_specialization_v<E1, ScalarZero> && is_specialization_v<E2, ScalarZero>) {
                return 0;
            }
            else if constexpr (is_specialization_v<E1, ScalarZero>) {
                return -m_expr2.eval();
            }
            else if constexpr (is_specialization_v<E2, ScalarZero>) {
                return m_expr1.eval();
            }
            else {
                return m_expr1.eval() - m_expr2.eval();
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ScalarExprType E1, ScalarExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E1& expr1, const E2& expr2) {
        return ScalarSubtraction<E1, E2>{expr1, expr2};
    }

    template<ScalarExprType E, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E& expr, S a) {
        return ScalarSubtraction<E, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ScalarExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(S a, const E& expr) {
        return ScalarSubtraction<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E1, class E2>
    class ScalarProduct : public AbstractScalarExpr<ScalarProduct<E1, E2>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarProduct(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarUnit> && is_specialization_v<decltype(expr2_derivative), ScalarUnit>) {
                return ScalarSum<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                    m_expr1,
                    m_expr2
                };
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero> && is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarZero<decltype(m_expr1.eval() * m_expr2.eval())>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero>) {
                return ScalarProduct<
                    std::remove_cvref_t<decltype(m_expr1)>,
                    decltype(expr2_derivative)
                >{
                    m_expr1,
                    expr2_derivative
                };
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarProduct<
                    decltype(expr1_derivative),
                    std::remove_cvref_t<decltype(m_expr2)>
                >{
                    expr1_derivative,
                    m_expr2
                };
            }
            else {
                return ScalarSum{
                    ScalarProduct<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    },
                    ScalarProduct<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                        expr1_derivative,
                        m_expr2
                    }
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            if constexpr (is_specialization_v<E1, ScalarZero> || is_specialization_v<E2, ScalarZero>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, ScalarUnit> && is_specialization_v<E2, ScalarUnit>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E1, ScalarUnit>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_specialization_v<E2, ScalarUnit>) {
                return m_expr1.to_string();
            }
            else {
                auto expr1_str = m_expr1.to_string();
                auto expr2_str = m_expr2.to_string();
                
                // Add parentheses around expressions that contain operators
                bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;
                
                if (expr1_needs_parens && expr2_needs_parens) {
                    return std::format("({}) * ({})", expr1_str, expr2_str);
                }
                else if (expr1_needs_parens) {
                    return std::format("({}) * {}", expr1_str, expr2_str);
                }
                else if (expr2_needs_parens) {
                    return std::format("{} * ({})", expr1_str, expr2_str);
                }
                else if (expr1_str.empty() || expr2_str.empty()) {
                    return std::format("{}{}", expr1_str, expr2_str);
                }
                else {
                    return std::format("{} * {}", expr1_str, expr2_str);
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            if constexpr (is_specialization_v<E1, ScalarZero> || is_specialization_v<E2, ScalarZero>) {
                return decltype(m_expr1.eval() * m_expr2.eval()){};
            }
            else if constexpr (is_specialization_v<E1, ScalarUnit> && is_specialization_v<E2, ScalarUnit>) {
                return static_cast<decltype(m_expr1.eval() * m_expr2.eval())>(1);
            }
            else if constexpr (is_specialization_v<E1, ScalarUnit>) {
                return m_expr2.eval();
            }
            else if constexpr (is_specialization_v<E2, ScalarUnit>) {
                return m_expr1.eval();
            }
            else {
                return m_expr1.eval() * m_expr2.eval();
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ScalarExprType E1, ScalarExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        return ScalarProduct<E1, E2>{expr1, expr2};
    }

    template<ScalarExprType E1, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr, S a) {
        return ScalarProduct<E1, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ScalarExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(S a, const E& expr) {
        return ScalarProduct<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E1, class E2>
    class ScalarDivision : public AbstractScalarExpr<ScalarDivision<E1, E2>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarDivision(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarUnit> && is_specialization_v<decltype(expr2_derivative), ScalarUnit>) {
                return ScalarSubtraction<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                    m_expr1,
                    m_expr2
                };
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero> && is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarZero<decltype(m_expr1.eval() * m_expr2.eval())>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ScalarZero>) {
                return ScalarProduct<
                    std::remove_cvref_t<decltype(m_expr1)>,
                    decltype(expr2_derivative)
                >{
                    m_expr1,
                    expr2_derivative
                };
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ScalarZero>) {
                return ScalarProduct<
                    decltype(expr1_derivative),
                    std::remove_cvref_t<decltype(m_expr2)>
                >{
                    expr1_derivative,
                    m_expr2
                };
            }
            else {
                return ScalarSum{
                    ScalarProduct<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    },
                    ScalarProduct<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                        expr1_derivative,
                        m_expr2
                    }
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline auto to_string() const {
            if constexpr (is_specialization_v<E1, ScalarZero>) {
                return "";
            }
            else if constexpr (is_specialization_v<E2, ScalarZero>) {
                throw std::runtime_error("[Division by zero in scalar expression.]");
            }
            else if constexpr (is_specialization_v<E1, ScalarUnit> && is_specialization_v<E2, ScalarUnit>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E2, ScalarUnit>) {
                return m_expr1.to_string();
            }
            else {
                auto expr1_str = m_expr1.to_string();
                auto expr2_str = m_expr2.to_string();
                
                // Add parentheses around expressions that contain operators
                bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('*') != std::string::npos || expr1_str.find('/') != std::string::npos;
                bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('*') != std::string::npos || expr2_str.find('/') != std::string::npos;

                if (expr1_needs_parens && expr2_needs_parens) {
                    return std::format("({}) / ({})", expr1_str, expr2_str);
                }
                else if (expr1_needs_parens) {
                    return std::format("({}) / {}", expr1_str, expr2_str);
                }
                else if (expr2_needs_parens) {
                    return std::format("{} / ({})", expr1_str, expr2_str);
                }
                else if (expr2_str.empty()) {
                    return std::format("[Division by zero in scalar expression.]");
                }
                else if (expr1_str.empty()) {
                    return std::format("");
                }
                else {
                    return std::format("{} / {}", expr1_str, expr2_str);
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            if constexpr (is_specialization_v<E2, ScalarZero>) {
                assert("Division by zero in scalar expression.");
            }
            else if constexpr (is_specialization_v<E2, ScalarUnit>) {
                return m_expr1.eval();
            }
            else {
                return m_expr1.eval() / m_expr2.eval();
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ScalarExprType E1, ScalarExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(const E1& expr1, const E2& expr2) {
        return ScalarDivision<E1, E2>{expr1, expr2};
    }

    template<ScalarExprType E1, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(const E1& expr, S a) {
        return ScalarDivision<E1, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ScalarExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(S a, const E& expr) {
        return ScalarDivision<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

}