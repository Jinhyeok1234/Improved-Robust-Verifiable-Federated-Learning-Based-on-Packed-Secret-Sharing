#include <iostream>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <thread>
#include <future>
#include <limits>
#include <atomic>
#include <cstring>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4146)
#endif

#include <gmpxx.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using BigInt = mpz_class;

// ---------------------------------------------------------------------------------
// Commitment group backends.
//
// The protocol needs a cyclic group of prime order q with two generators g and h. Z_p*
// with a prime-order subgroup is one instantiation; a prime-order elliptic curve is
// another. Shamir shares are elements of Z_q either way, so switching the backend leaves
// the share arithmetic, the serialization and the Python API untouched -- only the
// commitment arithmetic changes.
//
// A group element travels as a BigInt so that every container and serializer stays as it
// was. For the curve backend an affine point (x, y) is encoded as x * 2^fieldBits + y,
// with 0 reserved for the point at infinity ((0, 0) is on none of the supported curves).
//
// The curve backend is compiled only when VPSS_WITH_EC is defined; setup.py defines it
// when the OpenSSL headers are present.
// ---------------------------------------------------------------------------------

enum class GroupType { ZP = 0, EC = 1 };

#ifdef VPSS_WITH_EC
#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
#include <mutex>
#include <map>
#include <string>

namespace ecbackend {

inline BN_CTX* ctx() {
    thread_local BN_CTX* c = BN_CTX_new();      // BN_CTX is not thread safe; distribute() is OpenMP-parallel
    return c;
}

inline const EC_GROUP* group(int nid) {
    static std::mutex m;
    static std::map<int, EC_GROUP*> cache;      // EC_GROUP is read-only once built, so sharing is safe
    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find(nid);
    if (it != cache.end()) return it->second;
    EC_GROUP* g = EC_GROUP_new_by_curve_name(nid);
    if (!g) throw std::runtime_error("unknown curve nid");
    cache[nid] = g;
    return g;
}

inline BIGNUM* toBn(const BigInt& x) {
    size_t count = 0;
    std::vector<unsigned char> buf((mpz_sizeinbase(x.get_mpz_t(), 2) + 7) / 8 + 1);
    mpz_export(buf.data(), &count, 1, 1, 1, 0, x.get_mpz_t());
    return BN_bin2bn(buf.data(), static_cast<int>(count), nullptr);
}

inline BigInt fromBn(const BIGNUM* b) {
    BigInt out(0);
    int n = BN_num_bytes(b);
    if (n > 0) {
        std::vector<unsigned char> buf(static_cast<size_t>(n));
        BN_bn2bin(b, buf.data());
        mpz_import(out.get_mpz_t(), static_cast<size_t>(n), 1, 1, 1, 0, buf.data());
    }
    return out;
}

struct Point {
    const EC_GROUP* grp;
    EC_POINT* p;
    explicit Point(const EC_GROUP* g) : grp(g), p(EC_POINT_new(g)) {}
    ~Point() { if (p) EC_POINT_free(p); }
    Point(const Point&) = delete;
    Point& operator=(const Point&) = delete;
};

inline BigInt encode(int nid, unsigned fieldBits, const EC_POINT* pt) {
    const EC_GROUP* grp = group(nid);
    if (EC_POINT_is_at_infinity(grp, pt)) return BigInt(0);
    BIGNUM* x = BN_new();
    BIGNUM* y = BN_new();
    if (!EC_POINT_get_affine_coordinates(grp, pt, x, y, ctx())) {
        BN_free(x); BN_free(y);
        throw std::runtime_error("EC_POINT_get_affine_coordinates failed");
    }
    BigInt X = fromBn(x);
    BigInt Y = fromBn(y);
    BN_free(x); BN_free(y);
    return (X << fieldBits) + Y;
}

inline void decode(int nid, unsigned fieldBits, const BigInt& enc, EC_POINT* out) {
    const EC_GROUP* grp = group(nid);
    if (enc == 0) { EC_POINT_set_to_infinity(grp, out); return; }
    BigInt mask = (BigInt(1) << fieldBits) - 1;
    BigInt Y = enc & mask;
    BigInt X = enc >> fieldBits;
    BIGNUM* x = toBn(X);
    BIGNUM* y = toBn(Y);
    int ok = EC_POINT_set_affine_coordinates(grp, out, x, y, ctx());
    BN_free(x); BN_free(y);
    if (!ok) throw std::runtime_error("encoded value is not a point on the curve");
}

inline BigInt exp(int nid, unsigned fieldBits, const BigInt& base, const BigInt& e) {
    const EC_GROUP* grp = group(nid);
    Point b(grp), r(grp);
    decode(nid, fieldBits, base, b.p);
    BIGNUM* k = toBn(e);
    int ok = EC_POINT_mul(grp, r.p, nullptr, b.p, k, ctx());
    BN_free(k);
    if (!ok) throw std::runtime_error("EC_POINT_mul failed");
    return encode(nid, fieldBits, r.p);
}

inline BigInt mul(int nid, unsigned fieldBits, const BigInt& a, const BigInt& b) {
    const EC_GROUP* grp = group(nid);
    Point A(grp), B(grp), R(grp);
    decode(nid, fieldBits, a, A.p);
    decode(nid, fieldBits, b, B.p);
    if (!EC_POINT_add(grp, R.p, A.p, B.p, ctx())) throw std::runtime_error("EC_POINT_add failed");
    return encode(nid, fieldBits, R.p);
}

}  // namespace ecbackend
#endif  // VPSS_WITH_EC

using Polynomial = std::vector<BigInt>;
using LagrangeBasis = std::vector<Polynomial>;
using Matrix = std::vector<std::vector<BigInt>>;

enum class VerificationMode {
    None,
    Feldman,
    Pedersen
};

enum class PaddingMode {
    Zero,
    Random
};

/**
 * @brief Cryptographic public parameters for the VPSS scheme.
 */
struct PublicParameters {
    BigInt p; // Z_p backend: the large prime. EC backend: the field prime (informational).
    BigInt q; // Order of the group, where secrets and shares live
    BigInt g; // Generator g
    BigInt h; // Generator h (used for Pedersen)

    // Which instantiation of the commitment group is in use. ZP reproduces the original
    // behaviour exactly; EC carries affine points encoded as x * 2^fieldBits + y.
    GroupType groupType = GroupType::ZP;
    int curveNid = 0;          // EC only: OpenSSL curve NID
    unsigned fieldBits = 0;    // EC only: bits reserved for each affine coordinate
};

/**
 * @brief Represents a single share distributed to a user.
 */
template <VerificationMode V>
struct Share;

template <> struct Share<VerificationMode::Feldman> {
    BigInt x;   // User index (x-coordinate)
    BigInt s;   // Evaluation of secret polynomial f(x)
};
template <> struct Share<VerificationMode::None> {
    BigInt x;   // User index (x-coordinate)
    BigInt s;   // Evaluation of secret polynomial f(x)
};

template <> struct Share<VerificationMode::Pedersen> {
    BigInt x;   // User index (x-coordinate)
    BigInt s;   // Evaluation of secret polynomial f(x)
    BigInt r;   // Evaluation of blinding polynomial r(x) (0 if not Pedersen)
};

/**
 * @brief Represents a consolidated share for multiple secret blocks.
 * x-coordinate is stored once, while s and r are vectors.
 */
template <VerificationMode Mode>
struct BatchShare;

// Specialization for Feldman / None: Only s vector
template <> struct BatchShare<VerificationMode::Feldman> {
    BigInt x;
    std::vector<BigInt> s_values; // Vector of s shares for each block
};
template <> struct BatchShare<VerificationMode::None> {
    BigInt x;
    std::vector<BigInt> s_values;
};

// Specialization for Pedersen: s and r vectors
template <> struct BatchShare<VerificationMode::Pedersen> {
    BigInt x;
    std::vector<BigInt> s_values;
    std::vector<BigInt> r_values; // Vector of r shares for each block
};

/**
 * @brief Configuration object for initializing the VPSS scheme.
 */
struct VPSSConfig {
    // VerificationMode vssType;
    PaddingMode padType;
    bool deterministicRand = false;
    size_t t;                // Privacy threshold (reconstruction requires t shares)
    size_t k;                // Number of packed secrets
    size_t N;                // Total number of shares to generate

    std::vector<BigInt> secretIndices;  // X-coordinates where secrets are embedded
    std::vector<BigInt> shareIndices;   // X-coordinates where shares are evaluated

    bool isBasisCached = false;
    LagrangeBasis cachedSecretBasis, cachedShareBasis;
};

/**
 * @brief Provides mathematical utilities including modular arithmetic and polynomial interpolation.
 */
class MathHelper {
private:
    /**
     * @brief Helper function for recursive matrix chain multiplication.
     * * @param matrices The vector of matrices.
     * @param s The table containing optimal split indices.
     * @param i The start index.
     * @param j The end index.
     * @param mod The modulus for the field operations.
     * @return The result of multiplying matrices from index i to j.
     */
    static Matrix computeChainRecursive(const std::vector<Matrix>& matrices, 
                                      const std::vector<std::vector<int>>& s, 
                                      int i, int j, const BigInt& mod) {
        if (i == j) {
            return matrices[i];
        }
        
        int k = s[i][j];
        Matrix left = computeChainRecursive(matrices, s, i, k, mod);
        Matrix right = computeChainRecursive(matrices, s, k + 1, j, mod);
        
        return multiplyMatrix(left, right, mod);
    }
public:
    static BigInt randomBigInt(const BigInt& limit) {
        struct ThreadLocalGenerator {
            gmp_randclass engine;
            ThreadLocalGenerator() : engine(gmp_randinit_default) {
                auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
                unsigned long seed_val = static_cast<unsigned long>(now ^ tid);
                engine.seed(seed_val);
            }
        };
        thread_local ThreadLocalGenerator localGen;
        return localGen.engine.get_z_range(limit);
    }

    // --- commitment-group operations; the only place the backend matters ----------
    static BigInt groupIdentity(const PublicParameters& params) {
        return params.groupType == GroupType::ZP ? BigInt(1) : BigInt(0);
    }

    static BigInt groupExp(const PublicParameters& params, const BigInt& base, const BigInt& e) {
        if (params.groupType == GroupType::ZP) return modPow(base, e, params.p);
#ifdef VPSS_WITH_EC
        return ecbackend::exp(params.curveNid, params.fieldBits, base, e);
#else
        throw std::runtime_error("EC backend not compiled in (rebuild with VPSS_WITH_EC)");
#endif
    }

    static BigInt groupMul(const PublicParameters& params, const BigInt& a, const BigInt& b) {
        if (params.groupType == GroupType::ZP) return (a * b) % params.p;
#ifdef VPSS_WITH_EC
        return ecbackend::mul(params.curveNid, params.fieldBits, a, b);
#else
        throw std::runtime_error("EC backend not compiled in (rebuild with VPSS_WITH_EC)");
#endif
    }

    static BigInt modPow(const BigInt& base, const BigInt& exp, const BigInt& mod) {
        BigInt result;
        mpz_powm(result.get_mpz_t(), base.get_mpz_t(), exp.get_mpz_t(), mod.get_mpz_t());
        return result;
    }

    static BigInt modInverse(const BigInt& n, const BigInt& mod) {
        BigInt result;
        if (mpz_invert(result.get_mpz_t(), n.get_mpz_t(), mod.get_mpz_t()) == 0) {
            throw std::runtime_error("Modular inverse does not exist.");
        }
        return result;
    }

    /**
     * @brief Finds a generator 'g' of the subgroup of order q within Z_p*.
     * Assumes p = kq + 1. It picks a random h and computes g = h^k mod p.
     * If g == 1, it retries.
     */
    static BigInt findSubgroupGenerator(const BigInt& p, const BigInt& q) {
        BigInt p_minus_1 = p - 1;
        if (p_minus_1 % q != 0) {
            throw std::invalid_argument("q must divide p-1 (p = kq + 1).");
        }
        
        BigInt k = p_minus_1 / q;
        
        while (true) {
            // sample h randomly in [2, p-1]
            BigInt h = randomBigInt(p - 3) + 2; 
            
            // g = h^k mod p
            BigInt g = modPow(h, k, p);
            
            // g is a generator with order q if g != 1
            if (g != 1) {
                return g;
            }
        }
    }

    static BigInt evaluatePolynomial(const Polynomial& coeffs, const BigInt& x, const BigInt& mod) {
        if (coeffs.empty()) return 0;
        BigInt result = 0;
        for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; i--) {
            result = (result * x + coeffs[i]) % mod;
        }
        return result;
    }

    /**
     * @brief Precomputes Lagrange basis polynomials for a fixed set of x-coordinates.
     * @param xCoords Vector of x-coordinates.
     * @param mod The field modulus (order q).
     * @return A vector of polynomials, where basis[i] is the Lagrange basis for xCoords[i].
     */
    static LagrangeBasis precomputeBasis(const std::vector<BigInt>& xCoords, const BigInt& mod) {
        size_t n = xCoords.size();
        LagrangeBasis basis(n);

        for (size_t i = 0; i < n; ++i) {
            BigInt xi = xCoords[i];
            BigInt denominator = 1;
            Polynomial termCoeffs(1, 1);

            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                BigInt xj = xCoords[j];

                BigInt diff;
                if (xi >= xj) diff = xi - xj;
                else diff = mod - (xj - xi);
                denominator = (denominator * diff) % mod;

                Polynomial newTermCoeffs(termCoeffs.size() + 1, 0);
                BigInt negXj = (mod - xj) % mod;

                for (size_t k = 0; k < termCoeffs.size(); ++k) {
                    newTermCoeffs[k + 1] = (newTermCoeffs[k + 1] + termCoeffs[k]) % mod;
                    BigInt val = (termCoeffs[k] * negXj) % mod;
                    newTermCoeffs[k] = (newTermCoeffs[k] + val) % mod;
                }
                termCoeffs = newTermCoeffs;
            }

            BigInt denInv = modInverse(denominator, mod);
            for (auto& c : termCoeffs) {
                c = (c * denInv) % mod;
            }
            basis[i] = termCoeffs;
        }
        return basis;
    }

    /**
     * @brief Interpolates a polynomial passing through the given points.
     * @param points Vector of (x, y) pairs.
     * @param mod The field modulus.
     * @param basisCache Optional pointer to precomputed Lagrange basis.
     * @return Coefficients of the interpolated polynomial.
     */
    static Polynomial interpolatePolynomial(
        const std::vector<std::pair<BigInt, BigInt>>& points,
        const BigInt& mod,
        const LagrangeBasis* basisCache = nullptr
    ) {
        size_t n = points.size();

        LagrangeBasis tempBasis;
        const LagrangeBasis* basisPtr = basisCache;

        if (basisPtr == nullptr) {
            std::vector<BigInt> xCoords;
            xCoords.reserve(n);
            for (const auto& p : points) xCoords.push_back(p.first);

            tempBasis = precomputeBasis(xCoords, mod);
            basisPtr = &tempBasis;
        }

        Polynomial finalCoeffs(n, 0);

        for (size_t i = 0; i < n; ++i) {
            BigInt yi = points[i].second;
            const Polynomial& basisPoly = (*basisPtr)[i];

            // for (size_t k = 0; k < n; ++k) {
            //         BigInt term = (basisPoly[k] * yi) % mod;
            //         finalCoeffs[k] = (finalCoeffs[k] + term) % mod;
            //     }
            // }

            for (size_t k = 0; k < n; ++k) {
                BigInt term = basisPoly[k];
                term *= yi;
                term %= mod;

                finalCoeffs[k] += term; 
            }
        }

        for (size_t k = 0; k < n; ++k) {
            finalCoeffs[k] %= mod;
        }

        return finalCoeffs;
    }


    /**
     * @brief Solves a linear system Ax = b over GF(mod) using Gaussian elimination.
     * Needed for Berlekamp-Welch error correction.
     */
    static std::vector<BigInt> solveLinearSystem(std::vector<std::vector<BigInt>> A, std::vector<BigInt> b, const BigInt& mod) {
        size_t n = A.size();
        size_t m = A[0].size();
        
        // Initialize solution vector with 0 (Free variables will automatically be 0)
        std::vector<BigInt> solution(m, 0); 

        size_t r = 0; // Current row index being normalized
        std::vector<size_t> pivot_cols; // Record of column indices used as pivots for each row(r)

        // Iterate based on columns(c) and increment pivot row(r)
        for (size_t c = 0; c < m && r < n; ++c) {
            
            // 1. Find a non-zero pivot in the current column(c) modulo mod
            size_t pivot = r;
            while (pivot < n) {
                BigInt val = A[pivot][c] % mod;
                if (mpz_sgn(val.get_mpz_t()) != 0) break; // Non-zero value found
                pivot++;
            }

            // If all values in this column are 0, it's a free variable. Pass to the next column.
            if (pivot == n) {
                continue; 
            }

            // 2. Swap pivot row
            std::swap(A[r], A[pivot]);
            std::swap(b[r], b[pivot]);
            pivot_cols.push_back(c); // Record the column successfully used as a pivot

            // 3. Normalize pivot row (Make pivot 1)
            BigInt pivot_val = A[r][c] % mod;
            BigInt inv = modInverse(pivot_val, mod);
            for (size_t j = c; j < m; ++j) {
                A[r][j] = (A[r][j] * inv) % mod;
            }
            b[r] = (b[r] * inv) % mod;

            // 4. Eliminate all other rows (Gauss-Jordan)
            for (size_t k = 0; k < n; ++k) {
                if (k != r) {
                    BigInt factor = A[k][c] % mod;
                    if (mpz_sgn(factor.get_mpz_t()) != 0) {
                        for (size_t j = 0; j < m; ++j) {
                            BigInt sub = (factor * A[r][j]) % mod;
                            // Avoid negative values completely
                            A[k][j] = (A[k][j] + mod - sub) % mod;
                        }
                        BigInt subB = (factor * b[r]) % mod;
                        b[k] = (b[k] + mod - subB) % mod;
                    }
                }
            }
            
            // Move to the next row as elimination is complete
            r++; 
        }

        // 5. Map the calculated pivot column values to the solution vector
        for (size_t i = 0; i < pivot_cols.size(); ++i) {
            size_t c = pivot_cols[i];
            solution[c] = b[i] % mod;
        }

        return solution;
    }
    // static std::vector<BigInt> solveLinearSystem(std::vector<std::vector<BigInt>> A, std::vector<BigInt> b, const BigInt& mod) {
    //     size_t n = A.size();
    //     size_t m = A[0].size();

    //     for (size_t i = 0; i < std::min(n, m); ++i) {
    //         // Pivot selection
    //         size_t pivot = i;
    //         while (pivot < n && A[pivot][i] == 0) pivot++;
    //         if (pivot == n) continue; // No pivot in this column

    //         std::swap(A[i], A[pivot]);
    //         std::swap(b[i], b[pivot]);

    //         BigInt temp = A[i][i] % mod;
    //         if (mpz_sgn(temp.get_mpz_t()) == 0) {
    //             std::cerr << "[CRITICAL] Pivot is zero even after swap at row " << i << std::endl;
    //         }

    //         // Normalize pivot row
    //         BigInt inv = modInverse(A[i][i], mod);
    //         for (size_t j = i; j < m; ++j) A[i][j] = (A[i][j] * inv) % mod;
    //         b[i] = (b[i] * inv) % mod;

    //         // Eliminate other rows
    //         for (size_t k = 0; k < n; ++k) {
    //             if (k != i && A[k][i] != 0) {
    //                 BigInt factor = A[k][i];
    //                 for (size_t j = i; j < m; ++j) {
    //                     BigInt sub = (factor * A[i][j]) % mod;
    //                     BigInt val = A[k][j] - sub;
    //                     if (A[k][j] < sub) val += mod;
    //                     A[k][j] = val;
    //                 }
    //                 BigInt subB = (factor * b[i]) % mod;
    //                 BigInt val = b[k] - subB;
    //                 if (b[k] < subB) val += mod;
    //                 b[k] = val;
    //             }
    //         }
    // //     }

    //     // Extract solution (assuming unique solution for the first m variables)
    //     std::vector<BigInt> x(m);
    //     for (size_t i = 0; i < m; ++i) x[i] = b[i];
    //     return x;
    // }

    /**
     * @brief Performs Reed-Solomon Error Correction using the Berlekamp-Welch algorithm.
     * Reconstructs polynomial P(x) even if some shares are corrupted.
     * @param points All available shares (x, y).
     * @param degree The degree of the original polynomial (k + t - 1).
     * @param mod The field modulus.
     * @return The corrected polynomial coefficients.
     */
    static Polynomial berlekampWelchDecoding(
        const std::vector<std::pair<BigInt, BigInt>>& points,
        size_t degree,
        const BigInt& mod
    ) {
        size_t n = points.size();
        // Max errors e allowed: n >= degree + 1 + 2e
        size_t maxErrors = (n - 1 - degree) / 2;
        if (maxErrors == 0) return interpolatePolynomial(points, mod); // Fallback to simple interpolation

        // Berlekamp-Welch: Find E(x) and N(x) such that N(x_i) = y_i * E(x_i)
        // deg(E) = e, deg(N) = degree + e
        // E(x) = x^e + e_{e-1}x^{e-1} + ... + e_0 (monic)
        // Unknowns: coeffs of N (deg+e+1 vars) and coeffs of E (e vars) -> Total vars = deg + 2e + 1 <= n

        size_t e = maxErrors;
        size_t numVarsN = degree + e + 1;
        size_t numVarsE = e; // e_0 ... e_{e-1}
        size_t totalVars = numVarsN + numVarsE;

        std::vector<std::vector<BigInt>> Matrix(n, std::vector<BigInt>(totalVars, 0));
        std::vector<BigInt> B_vec(n, 0);

        for (size_t i = 0; i < n; ++i) {
            BigInt xi = points[i].first;
            BigInt yi = points[i].second;

            // N(xi) - yi * E(xi) = 0
            // N(xi) = n0 + n1*xi + ... 
            // -yi * E(xi) = -yi * (e0 + e1*xi + ... + xi^e)
            // => n0 + ... - yi*e0 - ... = yi * xi^e

            // Fill N coeffs part
            BigInt x_pow = 1;
            for (size_t j = 0; j < numVarsN; ++j) {
                Matrix[i][j] = x_pow;
                x_pow *= xi;
                x_pow %= mod;
            }

            // Fill E coeffs part (-yi * xi^j)
            x_pow = 1;
            for (size_t j = 0; j < numVarsE; ++j) {
                BigInt val = yi;
                val *= x_pow;
                val %= mod;
                // Subtract: (mod - val) % mod
                if (val == 0) Matrix[i][numVarsN + j] = 0;
                else Matrix[i][numVarsN + j] = mod - val;
                x_pow *= xi;
                x_pow %= mod;
            }

            // RHS: yi * xi^e
            B_vec[i] = yi;
            B_vec[i] *= x_pow;
            B_vec[i] %= mod;
        }

        std::vector<BigInt> solution = solveLinearSystem(Matrix, B_vec, mod);

        // Extract N(x) and E(x)
        Polynomial polyN(numVarsN);
        Polynomial polyE(e + 1);

        for (size_t i = 0; i < numVarsN; ++i) polyN[i] = solution[i];
        for (size_t i = 0; i < e; ++i) polyE[i] = solution[numVarsN + i];
        polyE[e] = 1; // Monic

        // P(x) = N(x) / E(x) (Polynomial division)
        // Since E(x) divides N(x) perfectly for valid RS codewords, 
        // we can just compute N(x)/E(x). 
        // Simple synthetic division or just evaluating at enough points and interpolating again.
        // Let's use simple synthetic division since E is monic.

        Polynomial polyP(degree + 1, 0);
        // polyN = polyP * polyE
        // Since E is monic, highest coeff of P is highest of N.
        for (int i = static_cast<int>(degree); i >= 0; --i) {
            // Prevent shallow copy references
            polyP[i] = (polyN[i + e] + 0) % mod; 
            for (int j = 0; j < static_cast<int>(e); ++j) {
                BigInt sub = (polyP[i] * polyE[j]) % mod;
                // Avoid negative values completely
                polyN[i + j] = (polyN[i + j] + mod - sub) % mod;
            }
        }

        return polyP;
    }

    /**
     * @brief Generates a Vandermonde matrix from a vector and a specified degree.
     * * Creates an N x (degree + 1) matrix where the i-th row corresponds to 
     * (1, x_i, x_i^2, ..., x_i^degree).
     * * @param vectorX The input vector containing N elements.
     * @param degree The maximum degree of the polynomial terms.
     * @param mod The modulus for the field operations.
     * @return A Matrix of size N x (degree + 1).
     */
    static Matrix createVandermondeMatrix(const std::vector<BigInt>& vectorX, size_t degree, const BigInt& mod) {
        size_t n = vectorX.size();
        size_t m = degree + 1;
        
        Matrix result(n, std::vector<BigInt>(m));

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            BigInt x = vectorX[i];
            BigInt val = 1;
            
            for (size_t j = 0; j < m; ++j) {
                result[i][j] = val;
                val = (val * x) % mod;
            }
        }
        return result;
    }

    /**
     * @brief Multiplies two matrices A and B modulo mod.
     * * Performs the operation C = A * B.
     * Uses OpenMP for parallel execution of row operations.
     * * @param A The left matrix (N x M).
     * @param B The right matrix (M x P).
     * @param mod The modulus for the field operations.
     * @return The product matrix C (N x P).
     * @throws std::runtime_error If matrix dimensions are incompatible.
     */
    static Matrix multiplyMatrix(const Matrix& A, const Matrix& B, const BigInt& mod) {
        if (A.empty() || B.empty()) {
            throw std::runtime_error("Matrix cannot be empty.");
        }
        
        size_t n = A.size();
        size_t m = A[0].size();
        size_t p = B[0].size();

        if (B.size() != m) {
            throw std::runtime_error("Dimension mismatch: Columns of A must match rows of B.");
        }

        Matrix C(n, std::vector<BigInt>(p, 0));

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            for (size_t j = 0; j < p; ++j) {
                BigInt sum = 0;
                for (size_t k = 0; k < m; ++k) {
                    BigInt term = (A[i][k] * B[k][j]) % mod;
                    sum = (sum + term) % mod;
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    /**
     * @brief Multiplies a matrix by a vector.
     * * Treats the input vector as a column vector (N x 1) and performs matrix multiplication.
     * * @param A The matrix (N x M).
     * @param vec The vector of length M.
     * @param mod The modulus for the field operations.
     * @return A Matrix representing the resulting column vector (N x 1).
     */
    static Matrix multiplyMatrix(const Matrix& A, const std::vector<BigInt>& vec, const BigInt& mod) {
        Matrix B(vec.size(), std::vector<BigInt>(1));
        for (size_t i = 0; i < vec.size(); ++i) {
            B[i][0] = vec[i];
        }
        return multiplyMatrix(A, B, mod);
    }

    /**
     * @brief Multiplies a vector by a matrix.
     * * Treats the input vector as a column vector (N x 1).
     * Note: This operation is mathematically valid only if the vector is treated 
     * as a row vector (1 x N) or if the dimensions align specifically for (N x 1) * (1 x M).
     * Based on the standard column-vector convention, this essentially converts the vector 
     * to a matrix and calls the matrix multiplication function.
     * * @param vec The vector of length N.
     * @param B The matrix.
     * @param mod The modulus for the field operations.
     * @return The result of the multiplication.
     */
    static Matrix multiplyMatrix(const std::vector<BigInt>& vec, const Matrix& B, const BigInt& mod) {
        Matrix A(vec.size(), std::vector<BigInt>(1));
        for (size_t i = 0; i < vec.size(); ++i) {
            A[i][0] = vec[i];
        }
        return multiplyMatrix(A, B, mod);
    }

    /**
     * @brief Computes the optimal matrix chain multiplication.
     * * Uses dynamic programming to determine the most efficient order of multiplication
     * to minimize the number of scalar operations.
     * * @param matrices A vector of matrices to be multiplied in sequence.
     * @param mod The modulus for the field operations.
     * @return The resulting matrix after all multiplications.
     * @throws std::runtime_error If no matrices are provided.
     */
    static Matrix multiplyMatrixChain(const std::vector<Matrix>& matrices, const BigInt& mod) {
        if (matrices.empty()) {
            throw std::runtime_error("Matrix list is empty.");
        }
        if (matrices.size() == 1) {
            return matrices[0];
        }

        size_t n = matrices.size();
        std::vector<size_t> p(n + 1);
        
        p[0] = matrices[0].size();
        for (size_t i = 0; i < n; ++i) {
            p[i+1] = matrices[i][0].size();
        }

        std::vector<std::vector<unsigned long long>> m(n, std::vector<unsigned long long>(n, 0));
        std::vector<std::vector<int>> s(n, std::vector<int>(n, 0));

        for (size_t len = 2; len <= n; ++len) {
            for (size_t i = 0; i <= n - len; ++i) {
                size_t j = i + len - 1;
                m[i][j] = std::numeric_limits<unsigned long long>::max();
                
                for (size_t k = i; k < j; ++k) {
                    unsigned long long cost = m[i][k] + m[k+1][j] + (p[i] * p[k+1] * p[j+1]);
                    if (cost < m[i][j]) {
                        m[i][j] = cost;
                        s[i][j] = static_cast<int>(k);
                    }
                }
            }
        }

        return computeChainRecursive(matrices, s, 0, static_cast<int>(n) - 1, mod);
    }

    /**
     * @brief Computes the modular inverse of a square matrix.
     * * Uses Gauss-Jordan elimination to transform [A | I] into [I | A^-1].
     * * @param A The square matrix to invert.
     * @param mod The modulus for the field operations.
     * @return The inverse matrix A^-1.
     * @throws std::runtime_error If the matrix is not square or is singular.
     */
    static Matrix invertMatrix(Matrix A, const BigInt& mod) {
        size_t n = A.size();
        if (n != A[0].size()) {
            throw std::runtime_error("Matrix must be square.");
        }

        Matrix I = createIdentityMatrix(n);

        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            while (pivot < n && A[pivot][i] == 0) {
                pivot++;
            }
            
            if (pivot == n) {
                throw std::runtime_error("Matrix is singular.");
            }

            if (i != pivot) {
                std::swap(A[i], A[pivot]);
                std::swap(I[i], I[pivot]);
            }

            BigInt inv = modInverse(A[i][i], mod);
            for (size_t j = 0; j < n; ++j) {
                A[i][j] = (A[i][j] * inv) % mod;
                I[i][j] = (I[i][j] * inv) % mod;
            }

            #pragma omp parallel for
            for (int k = 0; k < static_cast<int>(n); ++k) {
                if (static_cast<size_t>(k) != i) {
                    BigInt factor = A[k][i];
                    if (factor != 0) {
                        for (size_t j = 0; j < n; ++j) {
                            BigInt subA = (factor * A[i][j]) % mod;
                            BigInt val = A[k][j] - subA;
                            val = (val + mod) % mod;
                            A[k][j] = val;

                            BigInt subI = (factor * I[i][j]) % mod;
                            val = I[k][j] - subI;
                            val = (val + mod) % mod;
                            I[k][j] = val;
                        }
                    }
                }
            }
        }
        return I;
    }

    /**
     * @brief Creates an N x N identity matrix.
     * * @param n The dimension of the matrix.
     * @return The identity matrix.
     */
    static Matrix createIdentityMatrix(size_t n) {
        Matrix I(n, std::vector<BigInt>(n, 0));
        for (size_t i = 0; i < n; ++i) {
            I[i][i] = 1;
        }
        return I;
    }

    /**
     * @brief Creates an N x M matrix with random elements in Z_mod.
     * * @param mod The modulus defining the field.
     * @param n The number of rows.
     * @param m The number of columns.
     * @return A matrix with randomly sampled elements.
     */
    static Matrix createRandomMatrix(const BigInt& mod, size_t n, size_t m) {
        Matrix result(n, std::vector<BigInt>(m));
        
        gmp_randclass randGen(gmp_randinit_default);
        randGen.seed(static_cast<unsigned long>(time(NULL)));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                result[i][j] = randGen.get_z_range(mod);
            }
        }
        return result;
    }

    /**
     * @brief Creates a vector of length N with random elements in Z_mod.
     * * @param mod The modulus defining the field.
     * @param n The size of the vector.
     * @return A vector with randomly sampled elements.
     */
    static std::vector<BigInt> createRandomVector(const BigInt& mod, size_t n) {
        std::vector<BigInt> result(n);
        gmp_randclass randGen(gmp_randinit_default);
        randGen.seed(static_cast<unsigned long>(time(NULL)));

        for (size_t i = 0; i < n; ++i) {
            result[i] = randGen.get_z_range(mod);
        }
        return result;
    }

    /**
     * @brief Transposes a given matrix.
     * * @param A The matrix to transpose (N x M).
     * @return The transposed matrix A^T (M x N).
     */
    static Matrix transposeMatrix(const Matrix& A) {
        if (A.empty()) return Matrix();
        
        size_t rows = A.size();
        size_t cols = A[0].size();

        Matrix AT(cols, std::vector<BigInt>(rows));

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(rows); ++i) {
            for (size_t j = 0; j < cols; ++j) {
                AT[j][i] = A[i][j];
            }
        }
        return AT;
    }

    /**
     * @brief Computes sum(vec1[i] * vec2[i]) mod q efficiently in C++.
     */
    static BigInt computeDotProduct(
        const std::vector<BigInt>& vec1,
        const std::vector<BigInt>& vec2,
        const BigInt& mod
    ) {
        if (vec1.size() != vec2.size()) {
            throw std::invalid_argument("Vector dimensions do not match.");
        }

        BigInt total_sum = 0;
        size_t n = vec1.size();

        #pragma omp parallel
        {
            BigInt local_sum = 0;
            BigInt term;

            #pragma omp for
            for (int i = 0; i < static_cast<int>(n); ++i) {
                // term = vec1[i] * vec2[i]
                mpz_mul(term.get_mpz_t(), vec1[i].get_mpz_t(), vec2[i].get_mpz_t());
                // term = term % mod
                mpz_mod(term.get_mpz_t(), term.get_mpz_t(), mod.get_mpz_t());

                // local_sum = (local_sum + term) % mod
                mpz_add(local_sum.get_mpz_t(), local_sum.get_mpz_t(), term.get_mpz_t());
                mpz_mod(local_sum.get_mpz_t(), local_sum.get_mpz_t(), mod.get_mpz_t());
            }

            #pragma omp critical
            {
                mpz_add(total_sum.get_mpz_t(), total_sum.get_mpz_t(), local_sum.get_mpz_t());
                mpz_mod(total_sum.get_mpz_t(), total_sum.get_mpz_t(), mod.get_mpz_t());
            }
        }
        return total_sum;
    }

    /**
     * @brief Adds the elements of vectorB to vectorA in-place.
     * @param vectorA The destination vector, which will be modified to store the sum.
     * @param vectorB The source vector to add to vectorA.
     * @return void
     */
    static void addVectors(std::vector<BigInt>& vectorA, const std::vector<BigInt>& vectorB, const BigInt& mod) {
        if (vectorA.size() != vectorB.size()) {
            throw std::invalid_argument("Vector sizes must be equal.");
        }

        // [B7] The product / sum is written into a per-thread scratch mpz and only the reduced
        // result goes back into the destination. Writing the wider intermediate into the
        // destination itself made GMP reallocate that element on every iteration, and the
        // allocator lock then serialised the parallel loop: with a 9-limb modulus, adding a
        // second thread made this loop SLOWER rather than faster. The scratch is allocated
        // once per thread, wide enough for the intermediate, so nothing reallocates inside.
        const size_t scratch_bits = mpz_sizeinbase(mod.get_mpz_t(), 2) + 128;
        // [B7] Give every destination element room for a full residue BEFORE going parallel.
        // A share vector holds values of mixed width -- when k equals t the polynomial carries
        // no random coefficient, so some evaluations are only a few limbs wide -- and writing a
        // full-width residue into a narrow element reallocates it. Inside the parallel loop
        // that reallocation takes the allocator lock and makes the loop slower with more
        // threads than with one. Doing it here is serial, once per vector, and allocation-free
        // for elements that are already wide enough.
        const size_t n_elements = vectorA.size();
        const size_t mod_bits = mpz_sizeinbase(mod.get_mpz_t(), 2);
        const size_t mod_limbs = (mod_bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
        for (size_t index = 0; index < vectorA.size(); ++index) {
            if (mpz_size(vectorA[index].get_mpz_t()) < mod_limbs) {
                mpz_realloc2(vectorA[index].get_mpz_t(), mod_bits + GMP_NUMB_BITS);
            }
        }

#ifdef _OPENMP
        const long long CHUNK = std::max<long long>(
            256, static_cast<long long>(n_elements) / (8LL * omp_get_max_threads()));
#else
        const long long CHUNK = 256;
#endif

        #pragma omp parallel
        {
            mpz_t scratch;
            mpz_init2(scratch, scratch_bits);
            // [B7] Dynamic schedule with a chunk sized from the actual work. When k equals t
            // the polynomial carries no random coefficient, so a share vector mixes narrow and
            // full-width values and a static split stalls on the heavy chunk. A fixed chunk is
            // the wrong answer though: it has to be small enough that there are several chunks
            // per thread (otherwise dynamic behaves like static) and large enough that handing
            // one out costs less than doing it (otherwise the uniform case pays for nothing).
            // Aiming at 8 chunks per thread, floored at 256, satisfies both.
            #pragma omp for schedule(dynamic, CHUNK)
            for (long long index = 0; index < static_cast<long long>(vectorA.size()); ++index) {
                mpz_add(scratch, vectorA[index].get_mpz_t(), vectorB[index].get_mpz_t());
                mpz_mod(vectorA[index].get_mpz_t(), scratch, mod.get_mpz_t());
            }
            mpz_clear(scratch);
        }
    }

    /**
     * @brief Scales the elements of the vector by a given scalar in-place.
     * @param vector The vector to be scaled.
     * @param scalar The BigInt value to multiply with each element.
     * @return void
     */
    static void scaleVector(std::vector<BigInt>& vector, const BigInt& scalar, const BigInt& mod) {

        // [B7] Same reason as addVectors: keep the wide intermediate out of the destination.
        const size_t scratch_bits = mpz_sizeinbase(mod.get_mpz_t(), 2)
                                  + mpz_sizeinbase(scalar.get_mpz_t(), 2) + 128;
        // [B7] Give every destination element room for a full residue BEFORE going parallel.
        // A share vector holds values of mixed width -- when k equals t the polynomial carries
        // no random coefficient, so some evaluations are only a few limbs wide -- and writing a
        // full-width residue into a narrow element reallocates it. Inside the parallel loop
        // that reallocation takes the allocator lock and makes the loop slower with more
        // threads than with one. Doing it here is serial, once per vector, and allocation-free
        // for elements that are already wide enough.
        const size_t n_elements = vector.size();
        const size_t mod_bits = mpz_sizeinbase(mod.get_mpz_t(), 2);
        const size_t mod_limbs = (mod_bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
        for (size_t index = 0; index < vector.size(); ++index) {
            if (mpz_size(vector[index].get_mpz_t()) < mod_limbs) {
                mpz_realloc2(vector[index].get_mpz_t(), mod_bits + GMP_NUMB_BITS);
            }
        }

#ifdef _OPENMP
        const long long CHUNK = std::max<long long>(
            256, static_cast<long long>(n_elements) / (8LL * omp_get_max_threads()));
#else
        const long long CHUNK = 256;
#endif

        #pragma omp parallel
        {
            mpz_t scratch;
            mpz_init2(scratch, scratch_bits);
            // [B7] Dynamic schedule with a chunk sized from the actual work. When k equals t
            // the polynomial carries no random coefficient, so a share vector mixes narrow and
            // full-width values and a static split stalls on the heavy chunk. A fixed chunk is
            // the wrong answer though: it has to be small enough that there are several chunks
            // per thread (otherwise dynamic behaves like static) and large enough that handing
            // one out costs less than doing it (otherwise the uniform case pays for nothing).
            // Aiming at 8 chunks per thread, floored at 256, satisfies both.
            #pragma omp for schedule(dynamic, CHUNK)
            for (long long index = 0; index < static_cast<long long>(vector.size()); ++index) {
                mpz_mul(scratch, vector[index].get_mpz_t(), scalar.get_mpz_t());
                mpz_mod(vector[index].get_mpz_t(), scratch, mod.get_mpz_t());
            }
            mpz_clear(scratch);
        }
    }
};

/**
 * @brief Unified implementation of Shamir, Packed Shamir, Feldman VSS, and Pedersen VSS.
 */
template <VerificationMode Mode>
class UnifiedVPSS {
public:
    static void prepareBasisCache(
        const PublicParameters& params,
        VPSSConfig& config
    ) {
        if (config.isBasisCached)
            return;

        size_t t_actual = std::min(static_cast<size_t>(config.t), config.secretIndices.size());
        size_t n_actual = std::min(static_cast<size_t>(config.N), config.shareIndices.size());

        std::vector<BigInt> truncatedSecretIndices(
            config.secretIndices.begin(), 
            config.secretIndices.begin() + t_actual
        );
        std::vector<BigInt> truncatedShareIndices(
            config.shareIndices.begin(), 
            config.shareIndices.begin() + n_actual
        );

        config.cachedSecretBasis = MathHelper::precomputeBasis(truncatedSecretIndices, params.q);
        config.cachedShareBasis = MathHelper::precomputeBasis(truncatedShareIndices, params.q);
    }
    
    static Polynomial createPolynomialFromSecrets(
        const PublicParameters& params,
        const VPSSConfig& config, 
        const std::vector<BigInt>& secrets
    ){
        if (secrets.size() > config.k) {
            throw std::invalid_argument("Number of secrets should be equal or smaller than the packing parameter k.");
        }

        std::vector<std::pair<BigInt, BigInt>> points;

        for (size_t i = 0; i < secrets.size(); ++i) {
            points.push_back({ config.secretIndices[i], secrets[i] });
        }

        for (size_t i = secrets.size(); i < config.k; ++i) {
            BigInt padVal = (config.padType==PaddingMode::Zero) ? 0 : MathHelper::randomBigInt(params.q);
            points.push_back({ config.secretIndices[i], padVal });
        }

        for (size_t i = config.k; i < config.t; ++i) {
            BigInt randVal = config.deterministicRand ? BigInt(0) : MathHelper::randomBigInt(params.q);
            points.push_back({ config.secretIndices[i], randVal });
        }

        return MathHelper::interpolatePolynomial(points, params.q, &config.cachedSecretBasis);
    }

    /**
     * @brief Distributes secrets into shares and generates commitments.
     * @param secrets Vector of secrets to be packed. Size must match config.k.
     * @param outShares Output vector for generated shares.
     * @param outCommitments Output vector for polynomial coefficient commitments.
     */
    static void distribute(
        const PublicParameters& params,
        const VPSSConfig& config,
        const std::vector<BigInt>& secrets,
        std::vector<Share<Mode>>& outShares,
        std::vector<BigInt>& outCommitments,
        std::vector<BigInt>* outSecretBlinds = nullptr
    ) {
        if (secrets.size() > config.k) {
            throw std::invalid_argument("Number of secrets should be equal or smaller than the packing parameter k.");
        }

        Polynomial polyF = UnifiedVPSS::createPolynomialFromSecrets(params, config, secrets);

        Polynomial polyR;
        if constexpr (Mode == VerificationMode::Pedersen) {
            std::vector<BigInt> randomBlinds;
            randomBlinds.reserve(config.k);
            for (size_t i = 0; i < config.k; ++i) {
                randomBlinds.push_back(MathHelper::randomBigInt(params.q));
            }
            polyR = UnifiedVPSS::createPolynomialFromSecrets(params, config, randomBlinds);

            if (outSecretBlinds != nullptr) {
                outSecretBlinds->clear();
                outSecretBlinds->reserve(secrets.size());
                for (size_t i = 0; i < secrets.size(); ++i) {
                    BigInt evalPoint = config.secretIndices[i];
                    outSecretBlinds->push_back(MathHelper::evaluatePolynomial(polyR, evalPoint, params.q));
                }
            }
        }

        outShares.clear();
        for (size_t i = 0; i < config.N; ++i) {
            BigInt xCoord = config.shareIndices[i];
            BigInt sVal = MathHelper::evaluatePolynomial(polyF, xCoord, params.q);

            if constexpr (Mode == VerificationMode::Pedersen) {
                BigInt rVal = MathHelper::evaluatePolynomial(polyR, xCoord, params.q);
                outShares.push_back({ xCoord, sVal, rVal }); // {x, s, r}
            }
            else {
                outShares.push_back({ xCoord, sVal }); // {x, s}
            }
        }

        outCommitments.clear();
        if constexpr (Mode != VerificationMode::None) {
            for (size_t t = 0; t < polyF.size(); ++t) {
                BigInt g_a = MathHelper::groupExp(params, params.g, polyF[t]);
                BigInt commitment = g_a;
                if constexpr (Mode == VerificationMode::Pedersen) {
                    BigInt h_r = MathHelper::groupExp(params, params.h, polyR[t]);
                    commitment = MathHelper::groupMul(params, commitment, h_r);
                }
                outCommitments.push_back(commitment);
            }
        }
    }

    /**
     * @brief Verifies a single share against the public commitments.
     * @param share The share to verify.
     * @param commitments Public commitments for the polynomial coefficients.
     * @return true if the share is valid, false otherwise.
     */
    static bool verifyShare(
        const PublicParameters& params,
        const Share<Mode>& share,
        const std::vector<BigInt>& commitments
    ) {
        if constexpr (Mode == VerificationMode::None) return true;

        // Check: g^s * h^r == Product( C_i ^ (x^i) )

        BigInt lhs = MathHelper::groupExp(params, params.g, share.s);
        if constexpr (Mode == VerificationMode::Pedersen) {
            BigInt h_r = MathHelper::groupExp(params, params.h, share.r);
            lhs = MathHelper::groupMul(params, lhs, h_r);
        }

        BigInt rhs = MathHelper::groupIdentity(params);
        BigInt x_pow = 1;
        for (const auto& C : commitments) {
            BigInt term = MathHelper::groupExp(params, C, x_pow);
            rhs = MathHelper::groupMul(params, rhs, term);
            x_pow = (x_pow * share.x) % params.q;
        }
        return lhs == rhs;
    }

    /**
     * @brief Simple reconstruction using Lagrange Interpolation.
     * Fast, assumes provided shares are valid.
     * @param shares Subset of shares (must be at least size k+t).
     * @param verifyRemaining If true, checks if input shares lie on the same polynomial (Error Detection).
     */
    static std::vector<BigInt> reconstructSimple(
        const PublicParameters& params,
        const VPSSConfig& config, 
        const std::vector<Share<Mode>>& shares,
        bool verifyConsistency = false
    ) {
        size_t degree = config.t - 1;
        size_t minShares = degree + 1;

        if (shares.size() < minShares) {
            throw std::invalid_argument("Not enough shares for reconstruction.");
        }

        Polynomial poly;

        // [Optimization] If we have exactly N shares matching the cached indices, use the cache.
        // This avoids recomputing Lagrange basis (expensive modular inverses).
        bool canUseCache = false;
        if (config.isBasisCached && shares.size() == config.N) {
            canUseCache = true;
            // Verify indices match cachedShareBasis logic (assuming shares are sorted or matching shareIndices)
            for (size_t i = 0; i < config.N; ++i) {
                if (shares[i].x != config.shareIndices[i]) {
                    canUseCache = false;
                    break;
                }
            }
        }

        if (canUseCache) {
            // --- Method B: Interpolate N points using Cache ---
            std::vector<std::pair<BigInt, BigInt>> points;
            // Actually interpolatePolynomial with cache doesn't use points.first (x) for basis calculation, 
            // but we pass pairs to match signature.
            for (const auto& s : shares) points.push_back({ s.x, s.s });

            // This is O(N^2) but only additions/multiplications (No Inversions)
            poly = MathHelper::interpolatePolynomial(points, params.q, &config.cachedShareBasis);

            // Consistency Check: Ensure degree is effectively 'degree'
            // Coefficients from index [degree + 1] to [N - 1] must be 0
            if (verifyConsistency) {
                for (size_t i = degree + 1; i < poly.size(); ++i) {
                    if (poly[i] != 0) throw std::runtime_error("Share consistency check failed (Degree too high).");
                }
            }
        }
        else {
            // --- Method A: Subset Interpolation (Slow, requires basis re-computation) ---
            std::vector<std::pair<BigInt, BigInt>> points;
            for (size_t i = 0; i < minShares; ++i) points.push_back({ shares[i].x, shares[i].s });

            // Recomputes basis internally
            poly = MathHelper::interpolatePolynomial(points, params.q);

            if (verifyConsistency) {
                for (size_t i = minShares; i < shares.size(); ++i) {
                    BigInt y_calc = MathHelper::evaluatePolynomial(poly, shares[i].x, params.q);
                    if (y_calc != shares[i].s) throw std::runtime_error("Share inconsistency detected.");
                }
            }
        }

        // Recover secrets
        std::vector<BigInt> secrets;
        for (const auto& idx : config.secretIndices) {
            if (secrets.size() >= config.k) break;
            secrets.push_back(MathHelper::evaluatePolynomial(poly, idx, params.q));
        }
        return secrets;
    }

    /**
     * @brief Reconstruction with Reed-Solomon Error Correction.
     * Slower, but can recover secrets even if some shares are corrupted.
     */
    static std::vector<BigInt> reconstructWithCorrection(
        const PublicParameters& params,
        const VPSSConfig& config, 
        const std::vector<Share<Mode>>& shares
    ) {
        size_t degree = config.t - 1;

        std::vector<std::pair<BigInt, BigInt>> points;
        for (const auto& sh : shares) points.push_back({ sh.x, sh.s });

        // Use Berlekamp-Welch
        Polynomial poly = MathHelper::berlekampWelchDecoding(points, degree, params.q);

        std::vector<BigInt> secrets;
        for (const auto& idx : config.secretIndices) {
            if (secrets.size() >= config.k) break;
            secrets.push_back(MathHelper::evaluatePolynomial(poly, idx, params.q));
        }
        return secrets;
    }
};

/**
 * @brief Wrapper class to handle secrets longer than k by splitting them into blocks.
 * Uses UnifiedVPSS as the underlying engine.
 */
template <VerificationMode Mode>
class BatchVPSS {
public:
    /**
     * @brief Splits long secrets into chunks and distributes them.
     * @param longSecrets The complete vector of secrets (length > k allowed).
     * @param outBatchShares Output vector of consolidated BatchShares (size N).
     * @param outBatchCommitments Output matrix of commitments (BlockIndex -> Commitments).
     */
    static void distribute(
        const PublicParameters& params,
        const VPSSConfig& config,
        const std::vector<BigInt>& longSecrets,
        std::vector<BatchShare<Mode>>& outBatchShares,
        std::vector<std::vector<BigInt>>& outBatchCommitments,
        std::vector<BigInt>* outBatchSecretBlinds = nullptr,
        int threadCount=1
    ) {
        size_t totalSecrets = longSecrets.size();
        size_t numBlocks = (totalSecrets + config.k - 1) / config.k;

        // Initialize Outputs
        outBatchShares.resize(config.N);
        outBatchCommitments.resize(numBlocks);

        if (outBatchSecretBlinds != nullptr) {
            outBatchSecretBlinds->assign(totalSecrets, BigInt(0));
        }

        // Pre-allocate internal vectors for shares to avoid race conditions on push_back
        for (size_t i = 0; i < config.N; ++i) {
            // Assuming 1..N indices here. In real code, get from config.
            outBatchShares[i].x = config.shareIndices[i];
            outBatchShares[i].s_values.resize(numBlocks);
            if constexpr (Mode == VerificationMode::Pedersen) {
                outBatchShares[i].r_values.resize(numBlocks);
            }
        }

        // Launch Threads with OpenMP
        #pragma omp parallel for num_threads(threadCount)
        for (int b = 0; b < static_cast<int>(numBlocks); ++b) {
            // 1. Prepare Chunk
            std::vector<BigInt> chunk;
            chunk.reserve(config.k);
            size_t s_idx = b * config.k;
            size_t e_idx = std::min(s_idx + config.k, totalSecrets);
            for (size_t i = s_idx; i < e_idx; ++i) chunk.push_back(longSecrets[i]);

            // 2. Call Engine
            std::vector<Share<Mode>> tempShares;
            std::vector<BigInt> tempCommits;
            std::vector<BigInt> tempBlinds;

            std::vector<BigInt>* pTempBlinds = (outBatchSecretBlinds != nullptr) ? &tempBlinds : nullptr;

            UnifiedVPSS<Mode>::distribute(params, config, chunk, tempShares, tempCommits, pTempBlinds);

            // 3. Write results (Thread-safe due to disjoint indices)
            outBatchCommitments[b] = std::move(tempCommits);
            if (outBatchSecretBlinds != nullptr) {
                std::copy(tempBlinds.begin(), tempBlinds.end(), outBatchSecretBlinds->begin() + s_idx);
            }
            for (size_t i = 0; i < config.N; ++i) {
                outBatchShares[i].s_values[b] = std::move(tempShares[i].s);
                if constexpr (Mode == VerificationMode::Pedersen) {
                    outBatchShares[i].r_values[b] = std::move(tempShares[i].r);
                }
            }
        }
    }

    /**
     * @brief Verifies a batch share against all block commitments.
     */
    static bool verifyBatchShare(
        const PublicParameters& params,
        const BatchShare<Mode>& batchShare, 
        const std::vector<std::vector<BigInt>>& batchCommitments,
        int threadCount=1
    ) {
        if constexpr (Mode == VerificationMode::None) return true;

        size_t numBlocks = batchCommitments.size();
        
        // Basic size checks
        if (batchShare.s_values.size() != numBlocks) return false;
        if constexpr (Mode == VerificationMode::Pedersen) {
            if (batchShare.r_values.size() != numBlocks) return false;
        }

        // Shared flag for result (Atomic for thread safety)
        std::atomic<bool> allValid(true);

        #pragma omp parallel for num_threads(threadCount)
        for (int blockIdx = 0; blockIdx < static_cast<int>(numBlocks); ++blockIdx) {
            // Early exit if already failed (Relaxed read)
            if (!allValid.load(std::memory_order_relaxed)) continue;

            // Reconstruct a temporary single share
            Share<Mode> tempShare;
            tempShare.x = batchShare.x;
            tempShare.s = batchShare.s_values[blockIdx];
            
            if constexpr (Mode == VerificationMode::Pedersen) {
                tempShare.r = batchShare.r_values[blockIdx];
            }

            // Verify using the engine logic
            if (!UnifiedVPSS<Mode>::verifyShare(params, tempShare, batchCommitments[blockIdx])) {
                allValid.store(false, std::memory_order_relaxed);
            }
        }

        return allValid.load();
    }

    /**
     * @brief Parallelized reconstruction of batch shares.
     * @param batchShares Collected batch shares from t users.
     * @param useErrorCorrection If true, uses Berlekamp-Welch.
     */
    static std::vector<BigInt> reconstruct(
        const PublicParameters& params,
        const VPSSConfig& config, 
        const std::vector<BatchShare<Mode>>& batchShares,
        bool useErrorCorrection = false,
        int threadCount=1
    ) {
        if (batchShares.empty()) return {};
        size_t numBlocks = batchShares[0].s_values.size();
        size_t numUsers = batchShares.size();

        std::vector<BigInt> recoveredSecrets(numBlocks * config.k); // Flattened result

        #pragma omp parallel for num_threads(threadCount)
        for (int b = 0; b < static_cast<int>(numBlocks); ++b) {
            // Construct single block shares
            std::vector<Share<Mode>> blockShares;
            blockShares.reserve(numUsers);
            for (size_t u = 0; u < numUsers; ++u) {
                Share<Mode> s;
                s.x = batchShares[u].x;
                s.s = batchShares[u].s_values[b];
                blockShares.push_back(s);
            }

            std::vector<BigInt> blockResult;
            if (useErrorCorrection) {
                blockResult = UnifiedVPSS<Mode>::reconstructWithCorrection(params, config, blockShares);
            }
            else {
                blockResult = UnifiedVPSS<Mode>::reconstructSimple(params, config, blockShares);
            }

            for (size_t i = 0; i < blockResult.size(); ++i) {
                size_t globalIdx = b * config.k + i;
                if (globalIdx < recoveredSecrets.size()) {
                    recoveredSecrets[globalIdx] = blockResult[i];
                }
            }
        }

        // Remove potential padding from the end if needed is handled by vector size,
        // but here we just return the full vector including padding.
        // User logic determines real size.
        return recoveredSecrets;
    }
};