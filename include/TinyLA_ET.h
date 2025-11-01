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

    template<VarIDType varId, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto derivate(const E& expr) {
        static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
        return expr.derivate<varId>();
    }


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

    template<ExprType E1, ExprType E2>
    struct is_matrix_multiplicable {
        static constexpr bool value = (E1::cols == E2::rows);
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_matrix_multiplicable_v = is_matrix_multiplicable<E1, E2>::value;

    template<ScalarType T, uint32_t Row = 1, uint32_t Col = 1>
    class Zero : public AbstractExpr<Zero<T, Row, Col>, Row, Col> {
        public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Zero() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return Zero<T, Row, Col>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("0");
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return T{};
        }
    };

    // Specialized trait to check if a type is a Zero specialization
    template<class T>
    inline constexpr bool is_zero_v = false;

    template<ScalarType T, uint32_t Row, uint32_t Col>
    inline constexpr bool is_zero_v<Zero<T, Row, Col>> = true;

    template<ScalarType T, uint32_t N = 1>
    class Identity : public AbstractExpr<Identity<T, N>, N, N> {
        public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Identity() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return Zero<T, N, N>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (N == 1) {
                return std::string("1");
            }
            else {
                return std::string("I");
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            if constexpr (N == 1) {
                return T{1};
            }
            else {
                return (r == c) ? T{1} : T{};
            }
        }
    };

    // Specialized trait to check if a type is a Identity specialization
    template<class T>
    inline constexpr bool is_identity_v = false;

    template<ScalarType T, uint32_t N>
    inline constexpr bool is_identity_v<Identity<T, N>> = true;

    template<ScalarType T, uint32_t Row = 1, uint32_t Col = 1>
    class Ones : public AbstractExpr<Ones<T, Row, Col>, Row, Col> {
        public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Ones() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return Zero<T, Row, Col>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (Row == 1 && Col == 1) {
                return std::string("1");
            }
            else if constexpr (Row == 1) {
                std::stringstream strStream;
                strStream << "[ ";
                for (size_t c = 0; c < Col; ++c) {
                    strStream << "1";
                    if (c < Col - 1) {
                        strStream << " ";
                    }
                }
                return std::string(strStream << " ]");
            }
            else if constexpr (Col == 1) {
                std::stringstream strStream;
                for (size_t r = 0; r < Row; ++r) {
                    if (r < Row - 1) {
                        strStream << "| 1 |\n";
                    }
                    else {
                        strStream << "| 1 |";
                    }
                }
                return std::string(strStream.str());
            }
            else {
                std::stringstream strStream;
                for (size_t r = 0; r < Row; ++r) {
                    strStream << "| ";
                    for (size_t c = 0; c < Col; ++c) {
                        strStream << "1 ";
                    }
                    strStream << "|";
                    if (r < Row - 1) {
                        strStream << "\n";
                    }
                }
                return std::string(strStream);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
            return T{1};
        }

    };

    template<ScalarType T, uint32_t Row = 1, uint32_t Col = 1>
    class Constant : public AbstractExpr<Constant<T, Row, Col>, Row, Col> {
        public:

        CUDA_COMPATIBLE inline constexpr Constant(T value) : m_value(value) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return Zero<T, Row, Col>{};
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
    constexpr auto PI = Constant<T>{3.14159265358979323846264338327950288419716939937510582097494459230781640628};

    /*
        Euler number
    */
    template <ScalarType T>
    constexpr auto Euler = Constant<T>{2.718281828459045235360287471352662497757247093699959574966967627724076630353};

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

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Matrix(T value) : m_data{} {
            for (int r{}; r < Row; ++r) {
                for (int c{}; c < Col; ++c) {
                    m_data[r][c] = value;
                }
            }
        }

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractExpr<_SE>& expr) {
            static_assert(_SE::rows == Row && _SE::cols == Col, "Assigned expression must have the same dimensions as the matrix.");
            for (size_t r = 0; r < Row; ++r) {
                for (size_t c = 0; c < Col; ++c) {
                    m_data[r][c] = expr.eval(r, c);
                }
            }
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(T value) {
            for (int r{}; r < Row; ++r) {
                for (int c{}; c < Col; ++c) {
                    m_data[r][c] = value;
                }
            }
            return *this;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr operator T() const {
            static_assert(((*this).rows == 1 && (*this).cols == 1), "Only scalar shaped matrices can be assigned to scalar variables.");
            return m_data[0][0];
        }

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            if constexpr (diffVarId == varId) {
                return Ones<T, Row, Col>{};
            }
            else {
                return Zero<T, Row, Col>{};
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr ((*this).rows == 1 && (*this).cols == 1) {
                return (varId == -1)? std::format("{}", m_data[0][0]) : std::format("s_{}", varId);
            }
            else if constexpr ((*this).rows == 1) {
                std::stringstream strStream;
                strStream << "[";
                for (size_t c = 0; c < Col; ++c) {
                    if (c > 0) {
                        strStream << ", ";
                    }
                    strStream << m_data[0][c];
                }
                strStream << "]";
                return (varId == -1)? strStream.str() : std::format("v^T_{}", varId);
            }
            else if constexpr ((*this).cols == 1) {
                std::stringstream strStream;
                for (size_t r = 0; r < Row; ++r) {
                    strStream << '|' << m_data[r][0] << '|' << std::endl;
                }
                return (varId == -1)? strStream.str() : std::format("v_{}", varId);
            }
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
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


    // Type alias:
    template<ScalarType T, VarIDType varId = -1>
    using Scalar = Matrix<T, 1, 1, varId>;
    template<ScalarType T, uint32_t N, VarIDType varId = -1>
    using Vector = Matrix<T, N, 1, varId>;
    template<VarIDType varId = -1>
    using fscal = Matrix<float, 1, 1, varId>;
    template<VarIDType varId = -1>
    using fvec2 = Matrix<float, 2, 1, varId>;
    template<VarIDType varId = -1>
    using fvec3 = Matrix<float, 3, 1, varId>;
    template<VarIDType varId = -1>
    using fvec4 = Matrix<float, 4, 1, varId>;
    template<VarIDType varId = -1>
    using fmat2 = Matrix<float, 2, 2, varId>;
    template<VarIDType varId = -1>
    using fmat3 = Matrix<float, 3, 3, varId>;
    template<VarIDType varId = -1>
    using fmat4 = Matrix<float, 4, 4, varId>;
    template<VarIDType varId = -1>
    using dscal = Matrix<double, 1, 1, varId>;
    template<VarIDType varId = -1>
    using dvec2 = Matrix<double, 2, 1, varId>;
    template<VarIDType varId = -1>
    using dvec3 = Matrix<double, 3, 1, varId>;
    template<VarIDType varId = -1>
    using dvec4 = Matrix<double, 4, 1, varId>;
    template<VarIDType varId = -1>
    using dvec2 = Matrix<double, 2, 1, varId>;
    template<VarIDType varId = -1>
    using dmat2 = Matrix<double, 2, 2, varId>;
    template<VarIDType varId = -1>
    using dmat3 = Matrix<double, 3, 3, varId>;
    template<VarIDType varId = -1>
    using dmat4 = Matrix<double, 4, 4, varId>;
    template<VarIDType varId = -1>
    using cfscal = Matrix<std::complex<float>, 1, 1, varId>;
    template<VarIDType varId = -1>
    using cfvec2 = Matrix<std::complex<float>, 2, 1, varId>;
    template<VarIDType varId = -1>
    using cfvec3 = Matrix<std::complex<float>, 3, 1, varId>;
    template<VarIDType varId = -1>
    using cfvec4 = Matrix<std::complex<float>, 4, 1, varId>;
    template<VarIDType varId = -1>
    using cfmat2 = Matrix<std::complex<float>, 2, 2, varId>;
    template<VarIDType varId = -1>
    using cfmat3 = Matrix<std::complex<float>, 3, 3, varId>;
    template<VarIDType varId = -1>
    using cfmat4 = Matrix<std::complex<float>, 4, 4, varId>;
    template<VarIDType varId = -1>
    using cdscal = Matrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId = -1>
    using cdvec2 = Matrix<std::complex<double>, 2, 1, varId>;
    template<VarIDType varId = -1>
    using cdvec3 = Matrix<std::complex<double>, 3, 1, varId>;
    template<VarIDType varId = -1>
    using cdvec4 = Matrix<std::complex<double>, 4, 1, varId>;
    template<VarIDType varId = -1>
    using cdmat2 = Matrix<std::complex<double>, 2, 2, varId>;
    template<VarIDType varId = -1>
    using cdmat3 = Matrix<std::complex<double>, 3, 3, varId>;
    template<VarIDType varId = -1>
    using cdmat4 = Matrix<std::complex<double>, 4, 4, varId>;
    template<VarIDType varId = -1>
    using cscal = Matrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId = -1>
    using cscal = Matrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId = -1>
    using cvec2 = Matrix<std::complex<double>, 2, 1, varId>;
    template<VarIDType varId = -1>
    using cvec3 = Matrix<std::complex<double>, 3, 1, varId>;
    template<VarIDType varId = -1>
    using cmat2 = Matrix<std::complex<double>, 2, 2, varId>;
    template<VarIDType varId = -1>
    using cmat3 = Matrix<std::complex<double>, 3, 3, varId>;
    template<VarIDType varId = -1>
    using cmat4 = Matrix<std::complex<double>, 4, 4, varId>;

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
            if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                return Zero<decltype(m_expr1.eval(0, 0) + m_expr2.eval(0, 0)),
                    (*this).rows,
                    (*this).cols>{};
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                return expr2_derivative;
            }
            else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
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
            if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                return "";
            }
            else if constexpr (is_zero_v<E1>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_zero_v<E2>) {
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
            if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                return 0;
            }
            else if constexpr (is_zero_v<E1>) {
                return m_expr2.eval(r, c);
            }
            else if constexpr (is_zero_v<E2>) {
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
        return AdditionExpr<E, Constant<S>>{expr, Constant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(S a, const E& expr) {
        return AdditionExpr<Constant<S>, E>{Constant<S>{a}, expr};
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
            if constexpr (is_specialization_v<decltype(expr_derivative), Zero>) {
                return Zero<int>{};
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
            if constexpr (is_specialization_v<E, Zero>) {
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
            if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                return Zero<int>{};
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                return NegationExpr<decltype(expr2_derivative)>{expr2_derivative};
            }
            else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
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
            if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                return "";
            }
            else if constexpr (is_zero_v<E1>) {
                return std::format("-{}", m_expr2.to_string());
            }
            else if constexpr (is_zero_v<E2>) {
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
            if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                return 0;
            }
            else if constexpr (is_zero_v<E1>) {
                return -m_expr2.eval(r, c);
            }
            else if constexpr (is_zero_v<E2>) {
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
        return SubtractionExpr<E, Constant<S>>{expr, Constant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator-(S a, const E& expr) {
        return SubtractionExpr<Constant<S>, E>{Constant<S>{a}, expr};
    }

    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
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
            if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                    m_expr1,
                    m_expr2
                };
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                return Zero<decltype(m_expr1.eval(0, 0) * m_expr2.eval(0, 0))>{};
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                return ElementwiseProductExpr<
                    std::remove_cvref_t<decltype(m_expr1)>,
                    decltype(expr2_derivative)
                >{
                    m_expr1,
                    expr2_derivative
                };
            }
            else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
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
            if constexpr (is_specialization_v<E1, Zero> || is_specialization_v<E2, Zero>) {
                return "";
            }
            else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                return "1";
            }
            else if constexpr (is_identity_v<E1>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_identity_v<E2>) {
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
            if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                return decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c)){};
            }
            else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                return static_cast<decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c))>(1);
            }
            else if constexpr (is_identity_v<E1>) {
                return m_expr2.eval(r, c);
            }
            else if constexpr (is_identity_v<E2>) {
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

    template<ExprType E1, ExprType E2> requires(!is_matrix_multiplicable_v<E1, E2> && is_elementwise_broadcastable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        return ElementwiseProductExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E1, ScalarType S>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr, S a) {
        return ElementwiseProductExpr<E1, Constant<S>>{expr, Constant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(S a, const E& expr) {
        return ElementwiseProductExpr<Constant<S>, E>{Constant<S>{a}, expr};
    }

    template<ExprType E1, ExprType E2> requires(is_matrix_multiplicable_v<E1, E2>)
    class MatrixMultiplicationExpr : public AbstractExpr<MatrixMultiplicationExpr<E1, E2>,
        E1::rows,
        E2::cols
    > {
        public:

        CUDA_COMPATIBLE inline constexpr MatrixMultiplicationExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            static_assert(is_matrix_multiplicable_v<E1, E2>,
                "Incompatible matrix dimensions for matrix multiplication."
            );
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                    m_expr1,
                    m_expr2
                };
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                return Zero<decltype(m_expr1.eval(0, 0) * m_expr2.eval(0, 0)), (*this).rows, (*this).cols>{};
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                return MatrixMultiplicationExpr<
                    std::remove_cvref_t<decltype(m_expr1)>,
                    decltype(expr2_derivative)
                >{
                    m_expr1,
                    expr2_derivative
                };
            }
            else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                return MatrixMultiplicationExpr<
                    decltype(expr1_derivative),
                    std::remove_cvref_t<decltype(m_expr2)>
                >{
                    expr1_derivative,
                    m_expr2
                };
            }
            else {
                return AdditionExpr{
                    MatrixMultiplicationExpr<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                        expr1_derivative,
                        m_expr2
                    },
                    MatrixMultiplicationExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    }
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                return "";
            }
            else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                return "1";
            }
            else if constexpr (is_identity_v<E1>) {
                return m_expr2.to_string();
            }
            else if constexpr (is_identity_v<E2>) {
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
            if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                return decltype(m_expr1.eval(0, 0) * m_expr2.eval(0, 0)){};
            }
            else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                return Identity<decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c)), this->rows>{};
            }
            else if constexpr (is_identity_v<E1>) {
                return m_expr2.eval(r, c);
            }
            else if constexpr (is_identity_v<E2>) {
                return m_expr1.eval(r, c);
            }
            else {
                auto sum = decltype(m_expr1.eval(r, c) * m_expr2.eval(r, c)){};
                for (uint32_t k = 0; k < E1::cols; ++k)
                    sum += m_expr1.eval(r, k) * m_expr2.eval(k, c);
                auto temp = sum;
                return temp;
            }
        }

        private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_matrix_multiplicable_v<E1, E2>)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        return MatrixMultiplicationExpr<E1, E2>{expr1, expr2};
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
            if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                return Zero<decltype(m_expr1.eval(0, 0) / m_expr2.eval(0, 0))>{};
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
            if constexpr (is_zero_v<E1>) {
                return "";
            }
            else if constexpr (is_zero_v<E2>) {
                throw std::runtime_error("[Division by zero in scalar expression.]");
            }
            else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                return "1";
            }
            else if constexpr (is_identity_v<E2>) {
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
            if constexpr (is_zero_v<E2>) {
                throw std::runtime_error("Division by zero in scalar expression.");
            }
            else if constexpr (is_identity_v<E2>) {
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
        return DivisionExpr<E1, Constant<S>>{expr, Constant<S>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(S a, const E& expr) {
        return DivisionExpr<Constant<S>, E>{Constant<S>{a}, expr};
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
            if constexpr (is_specialization_v<decltype(expr_derivative), Zero>) {
                return Zero<decltype(m_expr.eval(0, 0))>{};
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
            if constexpr (is_zero_v<E>) {
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
            if constexpr (is_identity_v<E2>) {
                return expr1_derivative;
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>)
                 {
                return Zero<decltype(std::pow(m_expr1.eval(0, 0), m_expr2.eval(0, 0)))>{};
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
            if constexpr (is_zero_v<E1>) {
                return "";
            }
            else if constexpr (is_zero_v<E2>) {
                return "1";
            }
            else if constexpr (is_identity_v<E1>) {
                return "1";
            }
            else if constexpr (is_identity_v<E2>) {
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
        return ElementwisePowExpr<E, Constant<S>>{base, exponent};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto pow(S base, const E& exponent) {
        return ElementwisePowExpr<Constant<S>, E>{base, exponent};
    }

}