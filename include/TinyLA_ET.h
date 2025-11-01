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

    //-------------------------------------------------------------------------------

    // Trait to check if a type is a specialization of a template
    template <class, template<typename...> class>
    inline constexpr bool is_specialization_v = false;

    template <template<class...> class Template, class... Args>
    inline constexpr bool is_specialization_v<Template<Args...>, Template> = true;

    // Concepts:
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


    //---------------------------------------------------------------------------------------

    /*
        Variable ID type for automatic differentiation
    */
    using VarIDType = int16_t;

    template<class E, uint32_t Row = 1, uint32_t Col = 1>
    class AbstractExpr {
        public:
        static constexpr bool variable_data = false;
        static constexpr uint32_t rows = Row;
        static constexpr uint32_t cols = Col;

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            return static_cast<const E&>(*this).derivate<varId>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return static_cast<const E&>(*this).to_string();
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return (static_cast<const E&>(*this)).eval(r, c);
        }
    };

    template<class E>
    concept ExprType = requires(const E& e) {
        e.eval(0, 0);
        E::rows;
        E::cols;
        e.to_string();
    };

    template<ExprType E1, ExprType E2>
    struct is_eq_shape {
        static constexpr bool value = (E1::rows == E2::rows && E1::cols == E2::cols);
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_eq_shape_v = is_eq_shape<E1, E2>::value;

    template<ExprType E>
    struct is_scalar_shape {
        static constexpr bool value = (E::rows == 1 && E::cols == 1);
    };

    template<ExprType E>
    inline constexpr bool is_scalar_shape_v = is_scalar_shape<E>::value;

    template<ExprType E1, ExprType E2>
    struct is_elementwise_broadcastable {
        static constexpr bool value = is_eq_shape_v<E1, E2>
            || is_scalar_shape_v<E1> || is_scalar_shape_v<E2>;
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_elementwise_broadcastable_v = is_elementwise_broadcastable<E1, E2>::value;

    template<ScalarType T>
    class ZeroScalar : public AbstractExpr<ZeroScalar<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr ZeroScalar() {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");    
            return ZeroScalar<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("0");
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return T{};
        }
    };

    template<VarIDType varId, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto derivate(const E& expr) {
        static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
        return expr.derivate<varId>();
    }

    template<ScalarType T>
    class ScalarConstant : public AbstractExpr<ScalarConstant<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr ScalarConstant(T value) : m_value(value) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return ZeroScalar<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("{}", m_value);
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return m_value;
        }

        private:
        T m_value;
    };

    //---------------------------------------------------------------------------------------
    // Math constants:

    /*
        Pi constant
    */
    template <ScalarType T>
    constexpr auto PI = ScalarConstant<T>{3.14159265358979323846264338327950288419716939937510582097494459230781640628};

    /*
        Euler number
    */
    template <ScalarType T>
    constexpr auto Euler = ScalarConstant<T>{2.718281828459045235360287471352662497757247093699959574966967627724076630353};


    template<ScalarType T>
    class UnitScalar : public AbstractExpr<UnitScalar<T>> {
        public:

        CUDA_COMPATIBLE inline constexpr UnitScalar() {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return ZeroScalar<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("1");
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return static_cast<T>(1);
        }
    };

    template<ScalarType T, VarIDType varId = -1>
    class Scalar : public AbstractExpr<Scalar<T, varId>> {
        public:

        static constexpr bool variable_data = true;

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Scalar(const AbstractExpr<_SE>& expr) : m_value(expr.eval()) {}

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Scalar(T value = T{}) : m_value{value} {}

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractExpr<_SE>& expr) {
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
                return UnitScalar<T>();
            }
            else {
                return ZeroScalar<T>();
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return (varId == -1)? std::format("{}", m_value) : std::format("s_{}", varId);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return m_value;
        }

        private:
        T m_value;
    };

    template<ScalarType T, uint32_t Row, uint32_t Col, VarIDType varId = -1>
    class Matrix : public AbstractExpr<Matrix<T, Row, Col, varId>, Row, Col> {
        public:

        static constexpr bool variable_data = true;

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Matrix(const AbstractExpr<_SE>& expr) {
            for (size_t r = 0; r < Row; ++r) {
                for (size_t c = 0; c < Col; ++c) {
                    m_data[r][c] = expr.eval(r, c);
                }
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Matrix(const std::initializer_list<std::initializer_list<T>>& values) : m_data{} {
            size_t r = 0;
            for (const auto& row : values) {
                size_t c = 0;
                for (const auto& val : row) {
                    m_data[r][c] = val;
                    c++;
                }
                r++;
            }
        }

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractExpr<_SE>& expr) {
            m_data = expr.eval();
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(T value) {
            m_data = value;
            return *this;
        }

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            if constexpr (diffVarId == varId) {
                return UnitScalar<T>();
            }
            else {
                return ZeroScalar<T>();
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::stringstream strStream;
            for (size_t r = 0; r < Row; ++r) {
                for (size_t c = 0; c < Col; ++c) {
                    if (c > 0) {
                        strStream << ", ";
                    }
                    strStream << m_data[r][c];
                }
                strStream << std::endl;
            }
            return (varId == -1)? strStream.str() : std::format("M_{}", varId);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c) const {
            if constexpr (Row == 1 && Col == 1) {   // Behave as scalar
                return m_data[0][0];
            }
            if (r >= Row || c >= Col) {
                throw std::out_of_range("Matrix index out of range.");
            }
            return m_data[r][c];
        }

        private:
        T m_data[Row][Col];
    };

    template<ScalarType T, uint32_t N>
    class IdentityMatrix : public AbstractExpr<IdentityMatrix<T, N>, N, N> {
        public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr IdentityMatrix() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return ZeroScalar<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("I");
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c) const {
            return (r == c) ? T{1} : T{};
        }
    };

    template<ScalarType T, uint32_t Row, uint32_t Col>
    class ZeroMatrix : public AbstractExpr<ZeroMatrix<T, Row, Col>, Row, Col> {
        public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr ZeroMatrix() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return ZeroScalar<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("0");
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c) const {
            return T{};
        }
    };

    template<class E1, class E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    class AdditionExpr : public AbstractExpr<AdditionExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols>
    {
        public:

        CUDA_COMPATIBLE inline constexpr AdditionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_elementwise_broadcastable_v<E1, E2>,
                "Incompatible matrix dimensions for element-wise addition."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar> && is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return ZeroScalar<int>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar>) {
                return expr2_derivative;
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return expr1_derivative;
            }
            else {
                return AdditionExpr<
                    decltype(expr1_derivative),
                    decltype(expr2_derivative)
                >{
                    expr1_derivative,
                    expr2_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_specialization_v<E1, ZeroScalar> && is_specialization_v<E2, ZeroScalar>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (is_specialization_v<E1, ZeroScalar> && is_specialization_v<E2, ZeroScalar>) {
                return 0;
            }
            else if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return m_expr2.eval(r, c);
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
                return m_expr1.eval(r, c);
            }
            else {
                return m_expr1.eval(r, c) + m_expr2.eval(r, c);
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(const E1& expr1, const E2& expr2) {
        return AdditionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(const E& expr, S a) {
        return AdditionExpr<E, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(S a, const E& expr) {
        return AdditionExpr<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E>
    class NegationExpr : public AbstractExpr<NegationExpr<E>, E::rows, E::cols> {
        public:

        CUDA_COMPATIBLE inline constexpr NegationExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr_derivative), ZeroScalar>) {
                return ZeroScalar<int>{};
            }
            else {
                return NegationExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (is_specialization_v<E, ZeroScalar>) {
                return 0;
            }
            else {
                return -m_expr.eval(r, c);
            }
        }

        private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E& expr) {
        return NegationExpr<E>{expr};
    }

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    class SubtractionExpr : public AbstractExpr<SubtractionExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols 
    > {
        public:

        CUDA_COMPATIBLE inline constexpr SubtractionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_elementwise_broadcastable_v<E1, E2>,
                "Incompatible matrix dimensions for element-wise subtraction."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar> && is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return ZeroScalar<int>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar>) {
                return NegationExpr<decltype(expr2_derivative)>{expr2_derivative};
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return expr1_derivative;
            }
            else {
                return SubtractionExpr<
                    decltype(expr1_derivative),
                    decltype(expr2_derivative)
                >{
                    expr1_derivative,
                    expr2_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_specialization_v<E1, ZeroScalar> && is_specialization_v<E2, ZeroScalar>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return std::format("-{}", m_expr2.to_string());
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
                return m_expr1.to_string();
            }
            else {
                auto str1 = std::string(m_expr1.to_string());
                auto str2 = std::string(m_expr2.to_string());
                if (!str1.empty() && !str2.empty()) {
                    return std::format("{} - {}", str1, str2);
                }
                else if (!str1.empty()) {
                    return std::format("{}", str1);
                }
                else {
                    return std::format("-{}", str2);
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (is_specialization_v<E1, ZeroScalar> && is_specialization_v<E2, ZeroScalar>) {
                return 0;
            }
            else if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return -m_expr2.eval(r, c);
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
                return m_expr1.eval(r, c);
            }
            else {
                return m_expr1.eval(r, c) - m_expr2.eval(r, c);
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E1& expr1, const E2& expr2) {
        return SubtractionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(const E& expr, S a) {
        return SubtractionExpr<E, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(S a, const E& expr) {
        return SubtractionExpr<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E1, class E2>
    class ElementwiseProductExpr : public AbstractExpr<ElementwiseProductExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols
    > {
        public:

        CUDA_COMPATIBLE inline constexpr ElementwiseProductExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_elementwise_broadcastable_v<E1, E2>,
                "Incompatible matrix dimensions for element-wise product."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), UnitScalar> && is_specialization_v<decltype(expr2_derivative), UnitScalar>) {
                return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                    m_expr1,
                    m_expr2
                };
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar> && is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return ZeroScalar<decltype(m_expr1.eval() * m_expr2.eval())>{};
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar>) {
                return ElementwiseProductExpr<
                    std::remove_cvref_t<decltype(m_expr1)>,
                    decltype(expr2_derivative)
                >{
                    m_expr1,
                    expr2_derivative
                };
            }
            else if constexpr (is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return ElementwiseProductExpr<
                    decltype(expr1_derivative),
                    std::remove_cvref_t<decltype(m_expr2)>
                >{
                    expr1_derivative,
                    m_expr2
                };
            }
            else {
                return AdditionExpr{
                    ElementwiseProductExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    },
                    ElementwiseProductExpr<
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
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_specialization_v<E1, ZeroScalar> || is_specialization_v<E2, ZeroScalar>) {
                return "";
            }
            else if constexpr (is_specialization_v<E1, UnitScalar> && is_specialization_v<E2, UnitScalar>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E1, UnitScalar>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_specialization_v<E2, UnitScalar>) {
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
                    return std::format("");
                }
                else {
                    return std::format("{} * {}", expr1_str, expr2_str);
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (is_specialization_v<E1, ZeroScalar> || is_specialization_v<E2, ZeroScalar>) {
                return decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c)){};
            }
            else if constexpr (is_specialization_v<E1, UnitScalar> && is_specialization_v<E2, UnitScalar>) {
                return static_cast<decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c))>(1);
            }
            else if constexpr (is_specialization_v<E1, UnitScalar>) {
                return m_expr2.eval(r, c);
            }
            else if constexpr (is_specialization_v<E2, UnitScalar>) {
                return m_expr1.eval(r, c);
            }
            else {
                return m_expr1.eval(r, c) * m_expr2.eval(r, c);
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        return ElementwiseProductExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E1, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr, S a) {
        return ElementwiseProductExpr<E1, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(S a, const E& expr) {
        return ElementwiseProductExpr<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

    template<class E1, class E2>
    class DivisionExpr : public AbstractExpr<DivisionExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols
    > {
        public:

        CUDA_COMPATIBLE inline constexpr DivisionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_elementwise_broadcastable_v<E1, E2>,
                "Incompatible matrix dimensions for element-wise division."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar> && is_specialization_v<decltype(expr2_derivative), ZeroScalar>) {
                return ZeroScalar<decltype(m_expr1.eval() / m_expr2.eval())>{};
            }
            else {
                auto numerator = SubtractionExpr{
                    ElementwiseProductExpr<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                        expr1_derivative,
                        m_expr2
                    },
                    ElementwiseProductExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    }
                };
                auto denominator = ElementwiseProductExpr<
                    std::remove_cvref_t<decltype(m_expr2)>,
                    std::remove_cvref_t<decltype(m_expr2)>
                > {
                    m_expr2,
                    m_expr2
                };
                return DivisionExpr<decltype(numerator), decltype(denominator)>{
                    numerator,
                    denominator
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return "";
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
                throw std::runtime_error("[Division by zero in scalar expression.]");
            }
            else if constexpr (is_specialization_v<E1, UnitScalar> && is_specialization_v<E2, UnitScalar>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E2, UnitScalar>) {
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (is_specialization_v<E2, ZeroScalar>) {
                throw std::runtime_error("Division by zero in scalar expression.");
            }
            else if constexpr (is_specialization_v<E2, UnitScalar>) {
                return m_expr1.eval(r, c);
            }
            else {
                return m_expr1.eval(r, c) / m_expr2.eval(r, c);
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(const E1& expr1, const E2& expr2) {
        return DivisionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E1, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(const E1& expr, S a) {
        return DivisionExpr<E1, ScalarConstant<S>>{expr, ScalarConstant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(S a, const E& expr) {
        return DivisionExpr<ScalarConstant<S>, E>{ScalarConstant<S>{a}, expr};
    }

        template<class E>
    class NaturalLogExpr : public AbstractExpr<NaturalLogExpr<E>, E::rows, E::cols> {
        public:

        CUDA_COMPATIBLE inline constexpr NaturalLogExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_specialization_v<decltype(expr_derivative), ZeroScalar>) {
                return ZeroScalar<decltype(m_expr.eval())>{};
            }
            else {
                return DivisionExpr<
                    decltype(expr_derivative),
                    std::remove_cvref_t<decltype(m_expr)>
                >{
                    expr_derivative,
                    m_expr
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("[]Log of zero expression.]");
            }
            else {
                return std::format("log({})", str_expr);
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            auto expr_value = m_expr.eval(r, c);
            if constexpr (is_specialization_v<E, ZeroScalar>) {
                throw std::runtime_error("[Logarithm of zero in scalar expression.]");
            }
            else {
                return std::log(expr_value);
            }
        }

        private:
        std::conditional_t<(E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto log(const E& expr) {
        return NaturalLogExpr<E>{expr};
    }

    template<class E1, class E2>
    class ElementwisePowExpr : public AbstractExpr<ElementwisePowExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols>
    {
        public:

        CUDA_COMPATIBLE inline constexpr ElementwisePowExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_elementwise_broadcastable_v<E1, E2>,
                "Incompatible matrix dimensions for element-wise power operation."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_specialization_v<E2, UnitScalar>) {
                return expr1_derivative;
            }
            else if constexpr (is_specialization_v<decltype(expr1_derivative), ZeroScalar>)
                 {
                return ZeroScalar<decltype(std::pow(m_expr1.eval(), m_expr2.eval()))>{};
            }
            else {
                return ElementwiseProductExpr {
                    ElementwisePowExpr<E1, E2> {
                        m_expr1,
                        m_expr2
                    },
                    AdditionExpr {
                        ElementwiseProductExpr {
                            expr1_derivative,
                            DivisionExpr {
                                m_expr2,
                                m_expr1
                            }
                        },
                        ElementwiseProductExpr {
                            expr2_derivative,
                            NaturalLogExpr {
                                m_expr1
                            }
                        }
                    }
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_specialization_v<E1, ZeroScalar>) {
                return "";
            }
            else if constexpr (is_specialization_v<E2, ZeroScalar>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E1, UnitScalar>) {
                return "1";
            }
            else if constexpr (is_specialization_v<E2, UnitScalar>) {
                return m_expr1.to_string();
            }
            else {
                auto expr1_str = m_expr1.to_string();
                auto expr2_str = m_expr2.to_string();
                
                // Add parentheses around expressions that contain operators
                bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('*') != std::string::npos || expr1_str.find('/') != std::string::npos;
                bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('*') != std::string::npos || expr2_str.find('/') != std::string::npos;

                if (expr1_needs_parens && expr2_needs_parens) {
                    return std::format("({})^({})", expr1_str, expr2_str);
                }
                else if (expr1_needs_parens) {
                    return std::format("({})^{}", expr1_str, expr2_str);
                }
                else if (expr2_needs_parens) {
                    return std::format("{}^({})", expr1_str, expr2_str);
                }
                else if (expr2_str.empty()) {
                    return std::format("1");
                }
                else if (expr1_str.empty()) {
                    return std::format("");
                }
                else {
                    return std::format("{}^{}", expr1_str, expr2_str);
                }
            }
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return std::pow(m_expr1.eval(r, c), m_expr2.eval(r, c));
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto pow(const E1& base, const E2& exponent) {
        return ElementwisePowExpr<E1, E2>{base, exponent};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto pow(const E& base, S exponent) {
        return ElementwisePowExpr<E, ScalarConstant<S>>{base, exponent};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto pow(S base, const E& exponent) {
        return ElementwisePowExpr<ScalarConstant<S>, E>{base, exponent};
    }

}