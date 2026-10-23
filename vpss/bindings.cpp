#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include "VPSS.cpp" 

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<BigInt>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<BigInt>>);


using BatchSharePedersen = BatchShare<VerificationMode::Pedersen>;
using BatchShareFeldman = BatchShare<VerificationMode::Feldman>;
using BatchShareNone = BatchShare<VerificationMode::None>;

PYBIND11_MAKE_OPAQUE(std::vector<BatchSharePedersen>);
PYBIND11_MAKE_OPAQUE(std::vector<BatchShareFeldman>);
PYBIND11_MAKE_OPAQUE(std::vector<BatchShareNone>);

// -----------------------------------------------------------------------------
// Serialization Helpers (Binary Packing)
// -----------------------------------------------------------------------------
class Serializer {
    std::ostringstream oss;
public:
    Serializer() : oss(std::ios::binary) {}

    void write_bigint(const BigInt& n) {
        // [B1] Raw big-endian magnitude with a sign byte and a length prefix.
        // The previous encoding wrote the value as a hexadecimal ASCII string, so every
        // field element took two bytes per byte of value: a 512-bit residue went on the
        // wire as 128 characters plus a 4-byte length (132 bytes) instead of 69.
        uint8_t sign_byte = (mpz_sgn(n.get_mpz_t()) < 0) ? 1 : 0;
        size_t byte_count = 0;
        std::vector<unsigned char> buf((mpz_sizeinbase(n.get_mpz_t(), 2) + 7) / 8 + 1);
        mpz_export(buf.data(), &byte_count, 1, 1, 1, 0, n.get_mpz_t());
        uint32_t len = static_cast<uint32_t>(byte_count);
        oss.write(reinterpret_cast<const char*>(&sign_byte), 1);
        oss.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) oss.write(reinterpret_cast<const char*>(buf.data()), len);
    }

    template <typename T>
    void write_vector_bigint(const std::vector<T>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& item : vec) {
            write_bigint(item);
        }
    }

    void write_matrix_bigint(const std::vector<std::vector<BigInt>>& mat) {
        uint32_t rows = static_cast<uint32_t>(mat.size());
        oss.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        for (const auto& row : mat) {
            uint32_t cols = static_cast<uint32_t>(row.size());
            oss.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            for (const auto& val : row) {
                write_bigint(val);
            }
        }
    }

    // For BatchSharePedersen
    void write_share_pedersen(const BatchShare<VerificationMode::Pedersen>& share) {
        write_bigint(share.x);
        write_vector_bigint(share.s_values);
        write_vector_bigint(share.r_values);
    }
    void write_vector_share_pedersen(const std::vector<BatchSharePedersen>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& item : vec) {
            write_share_pedersen(item);
        }
    }

    // For BatchShareFeldman
    void write_share_feldman(const BatchShare<VerificationMode::Feldman>& share) {
        write_bigint(share.x);
        write_vector_bigint(share.s_values);
    }
    void write_vector_share_feldman(const std::vector<BatchShareFeldman>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& item : vec) {
            write_share_feldman(item);
        }
    }

    void write_share_none(const BatchShare<VerificationMode::None>& share) {
        write_bigint(share.x);
        write_vector_bigint(share.s_values);
    }
    void write_vector_share_none(const std::vector<BatchShareNone>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& item : vec) {
            write_share_none(item);
        }
    }



    std::string str() const { return oss.str(); }
    
    py::bytes to_bytes() const {
        std::string s = oss.str();
        return py::bytes(s);
    }
};

class Deserializer {
    std::string data;
    size_t pos = 0;
public:
    Deserializer(const std::string& s) : data(s), pos(0) {}

    BigInt read_bigint() {
        // [B1] counterpart of write_bigint: sign byte, length, big-endian magnitude.
        if (pos + 1 + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        uint8_t sign_byte = static_cast<uint8_t>(data[pos]);
        pos += 1;
        uint32_t len;
        std::memcpy(&len, &data[pos], sizeof(len));
        pos += sizeof(len);

        if (pos + len > data.size()) throw std::runtime_error("Stream too short for BigInt body");
        BigInt out(0);
        if (len > 0) {
            mpz_import(out.get_mpz_t(), len, 1, 1, 1, 0, &data[pos]);
        }
        pos += len;
        if (sign_byte) out = -out;
        return out;
    }

    std::vector<BigInt> read_vector_bigint() {
        if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        uint32_t size;
        std::memcpy(&size, &data[pos], sizeof(size));
        pos += sizeof(size);

        std::vector<BigInt> vec;
        vec.reserve(size);
        for (uint32_t i = 0; i < size; ++i) {
            vec.push_back(read_bigint());
        }
        return vec;
    }

    std::vector<std::vector<BigInt>> read_matrix_bigint() {
        uint32_t rows;
        if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        std::memcpy(&rows, &data[pos], sizeof(rows));
        pos += sizeof(rows);

        std::vector<std::vector<BigInt>> mat(rows);
        for (uint32_t i = 0; i < rows; ++i) {
            uint32_t cols;
            if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
            std::memcpy(&cols, &data[pos], sizeof(cols));
            pos += sizeof(cols);
            
            mat[i].reserve(cols);
            for (uint32_t j = 0; j < cols; ++j) {
                mat[i].push_back(read_bigint());
            }
        }
        return mat;
    }

    void read_share_pedersen(BatchShare<VerificationMode::Pedersen>& share) {
        share.x = read_bigint();
        share.s_values = read_vector_bigint();
        share.r_values = read_vector_bigint();
    }
    std::vector<BatchSharePedersen> read_vector_share_pedersen() {
        if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        uint32_t size;
        std::memcpy(&size, &data[pos], sizeof(size));
        pos += sizeof(size);
        
        std::vector<BatchSharePedersen> vec;
        vec.reserve(size);
        for(uint32_t i=0; i<size; ++i) {
            BatchSharePedersen share;
            read_share_pedersen(share);
            vec.push_back(share);
        }
        return vec;
    }

    void read_share_feldman(BatchShare<VerificationMode::Feldman>& share) {
        share.x = read_bigint();
        share.s_values = read_vector_bigint();
    }
    std::vector<BatchShareFeldman> read_vector_share_feldman() {
        if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        uint32_t size;
        std::memcpy(&size, &data[pos], sizeof(size));
        pos += sizeof(size);
        
        std::vector<BatchShareFeldman> vec;
        vec.reserve(size);
        for(uint32_t i=0; i<size; ++i) {
            BatchShareFeldman share;
            read_share_feldman(share);
            vec.push_back(share);
        }
        return vec;
    }

    void read_share_none(BatchShare<VerificationMode::None>& share) {
        share.x = read_bigint();
        share.s_values = read_vector_bigint();
    }
    std::vector<BatchShareNone> read_vector_share_none() {
        if (pos + sizeof(uint32_t) > data.size()) throw std::runtime_error("Stream too short");
        uint32_t size;
        std::memcpy(&size, &data[pos], sizeof(size));
        pos += sizeof(size);
        
        std::vector<BatchShareNone> vec;
        vec.reserve(size);
        for(uint32_t i=0; i<size; ++i) {
            BatchShareNone share;
            read_share_none(share);
            vec.push_back(share);
        }
        return vec;
    }
};

template <VerificationMode Mode>
class PyBatchVPSS {
public:
    static py::object
    distribute(const PublicParameters& params, 
               VPSSConfig& config, 
               const std::vector<BigInt>& secrets, 
               int thread_count,
               bool return_blinds = false) {
        
        UnifiedVPSS<Mode>::prepareBasisCache(params, config);

        std::vector<BatchShare<Mode>> shares;
        std::vector<std::vector<BigInt>> commits;
        std::vector<BigInt> blinds;

        std::vector<BigInt>* pBlinds = (return_blinds && Mode == VerificationMode::Pedersen) ? &blinds : nullptr;
        
        BatchVPSS<Mode>::distribute(params, config, secrets, shares, commits, pBlinds, thread_count);
        
        if (return_blinds) {
            return py::make_tuple(std::move(shares), std::move(commits), std::move(blinds));
        }
        return py::make_tuple(std::move(shares), std::move(commits));
    }

    static bool verify_batch_share(const PublicParameters& params,
                                   const BatchShare<Mode>& share,
                                   const std::vector<std::vector<BigInt>>& commitments,
                                   int thread_count) {
        return BatchVPSS<Mode>::verifyBatchShare(params, share, commitments, thread_count);
    }

    static std::vector<BigInt> reconstruct(const PublicParameters& params,
                                           VPSSConfig& config, 
                                           const std::vector<BatchShare<Mode>>& batchShares, 
                                           bool useErrorCorrection,
                                           int thread_count) {
        UnifiedVPSS<Mode>::prepareBasisCache(params, config);
        
        return BatchVPSS<Mode>::reconstruct(params, config, batchShares, useErrorCorrection, thread_count);
    }
};

PYBIND11_MODULE(vpss_module, m) {
    m.doc() = "VPSS Module";

    m.def("find_generator", &MathHelper::findSubgroupGenerator, 
          "Finds a generator of order q in Z_p*",
          py::arg("p"), py::arg("q"));

    m.def("random_bigint", &MathHelper::randomBigInt, 
          "Generates a random BigInt in range [0, limit-1]",
          py::arg("limit"));

          
    m.def("compute_dot_product", &MathHelper::computeDotProduct,
          "Computes sum(vec1[i] * vec2[i]) % q",
          py::arg("vec1"), py::arg("vec2"), py::arg("mod")
        );

    m.def("addVectors", &MathHelper::addVectors,
          "Adds the elements of vector2 to vector1 in-place.",
          py::arg("vec1"), py::arg("vec2"), py::arg("mod"));

    m.def("scaleVector", &MathHelper::scaleVector,
          "Scales the elements of the vector by a given scalar in-place.",
          py::arg("vec"), py::arg("scaler"), py::arg("mod"));

    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------
    py::enum_<VerificationMode>(m, "VerificationMode")
        .value("None", VerificationMode::None)
        .value("Feldman", VerificationMode::Feldman)
        .value("Pedersen", VerificationMode::Pedersen)
        .export_values();

    py::enum_<PaddingMode>(m, "PaddingMode")
        .value("Zero", PaddingMode::Zero)
        .value("Random", PaddingMode::Random)
        .export_values();

    // -------------------------------------------------------------------------
    // BigInt Wrapper
    // -------------------------------------------------------------------------
    py::class_<BigInt>(m, "BigInt")
        .def(pybind11::init([](long long v) {
            return mpz_class(static_cast<long>(v));
        }))
        .def(py::init([](std::string s) { return BigInt(s); }))
        .def("to_int", [](const BigInt& self) {
            py::object builtins = py::module_::import("builtins");
            return builtins.attr("int")(self.get_str());
        })
        .def("__str__", [](const BigInt& self) { return self.get_str(); })
        .def("__repr__", [](const BigInt& self) { return "<BigInt " + self.get_str() + ">"; })

        // .def(py::self + py::self)
        // .def(py::self - py::self)
        // .def(py::self * py::self)
        // .def(py::self / py::self)
        // .def(py::self % py::self)
        // .def(py::self == py::self)
        // .def(py::self != py::self)

        // .def("__add__", [](const BigInt& a, long long b) { return a + BigInt(static_cast<long>(b)); })
        // .def("__sub__", [](const BigInt& a, long long b) { return a - BigInt(static_cast<long>(b)); })
        // .def("__mul__", [](const BigInt& a, long long b) { return a * BigInt(static_cast<long>(b)); })
        // .def("__mod__", [](const BigInt& a, long long b) { return a % BigInt(static_cast<long>(b)); })
        // .def("__radd__", [](const BigInt& a, long long b) { return BigInt(static_cast<long>(b)) + a; })
        // .def("__rmul__", [](const BigInt& a, long long b) { return BigInt(static_cast<long>(b)) * a; })

        .def("__add__", [](const BigInt& a, const BigInt& b) -> BigInt { return BigInt(a + b); })
        .def("__add__", [](const BigInt& a, long long b) -> BigInt { return BigInt(a + mpz_class(static_cast<long>(b))); })
        .def("__radd__", [](const BigInt& a, long long b) -> BigInt { return BigInt(mpz_class(static_cast<long>(b)) + a); })

        .def("__sub__", [](const BigInt& a, const BigInt& b) -> BigInt { return BigInt(a - b); })
        .def("__sub__", [](const BigInt& a, long long b) -> BigInt { return BigInt(a - mpz_class(static_cast<long>(b))); })
        .def("__rsub__", [](const BigInt& a, long long b) -> BigInt { return BigInt(mpz_class(static_cast<long>(b)) - a); })

        .def("__mul__", [](const BigInt& a, const BigInt& b) -> BigInt { return BigInt(a * b); })
        .def("__mul__", [](const BigInt& a, long long b) -> BigInt { return BigInt(a * mpz_class(static_cast<long>(b))); })
        .def("__rmul__", [](const BigInt& a, long long b) -> BigInt { return BigInt(mpz_class(static_cast<long>(b)) * a); })

        .def("__truediv__", [](const BigInt& a, const BigInt& b) -> BigInt { return BigInt(a / b); })
        .def("__truediv__", [](const BigInt& a, long long b) -> BigInt { return BigInt(a / mpz_class(static_cast<long>(b))); })
        .def("__rtruediv__", [](const BigInt& a, long long b) -> BigInt { return BigInt(mpz_class(static_cast<long>(b)) / a); })

        .def("__mod__", [](const BigInt& a, const BigInt& b) -> BigInt { 
            return BigInt(((a % b) + b) % b); 
        })
        .def("__mod__", [](const BigInt& a, long long b) -> BigInt { 
            mpz_class b_mpz(static_cast<long>(b));
            return BigInt(((a % b_mpz) + b_mpz) % b_mpz); 
        })
        .def("__rmod__", [](const BigInt& a, long long b) -> BigInt { 
            mpz_class b_mpz(static_cast<long>(b));
            return BigInt(((b_mpz % a) + a) % a); 
        })

        .def("__eq__", [](const BigInt& a, const BigInt& b) { return a == b; })
        .def("__eq__", [](const BigInt& a, long long b) { return a == mpz_class(static_cast<long>(b)); })

        .def("__ne__", [](const BigInt& a, const BigInt& b) { return a != b; })
        .def("__ne__", [](const BigInt& a, long long b) { return a != mpz_class(static_cast<long>(b)); })

        .def("serialize", [](const BigInt& self) {
            Serializer s; s.write_bigint(self); return s.to_bytes();
        })
        .def_static("deserialize", [](py::bytes b) {
            Deserializer d(b); return d.read_bigint();
        })

        .def(py::pickle(
            [](const BigInt& self) { // __getstate__
                Serializer s; s.write_bigint(self); return s.to_bytes();
            },
            [](py::bytes state) { // __setstate__
                Deserializer d(state); return d.read_bigint();
            }
        ));

    // -------------------------------------------------------------------------
    // Opaque Vectors
    // -------------------------------------------------------------------------
    py::bind_vector<std::vector<BigInt>>(m, "BigIntVector")
        .def(py::pickle(
            [](const std::vector<BigInt>& self) {
                Serializer s; s.write_vector_bigint(self); return s.to_bytes();
            },
            [](py::bytes state) {
                Deserializer d(state); return d.read_vector_bigint();
            }
        ));
    py::bind_vector<std::vector<std::vector<BigInt>>>(m, "BigIntMatrix")
        .def(py::pickle(
            [](const std::vector<std::vector<BigInt>>& self) {
                Serializer s; s.write_matrix_bigint(self); return s.to_bytes();
            },
            [](py::bytes state) {
                Deserializer d(state); return d.read_matrix_bigint();
            }
        ));

    // -------------------------------------------------------------------------
    // Structures
    // -------------------------------------------------------------------------
    py::enum_<GroupType>(m, "GroupType")
        .value("ZP", GroupType::ZP)
        .value("EC", GroupType::EC)
        .export_values();

    py::class_<PublicParameters>(m, "PublicParameters")
        .def(py::init<>())
        .def_readwrite("p", &PublicParameters::p)
        .def_readwrite("q", &PublicParameters::q)
        .def_readwrite("g", &PublicParameters::g)
        .def_readwrite("h", &PublicParameters::h)
        .def_readwrite("groupType", &PublicParameters::groupType)
        .def_readwrite("curveNid", &PublicParameters::curveNid)
        .def_readwrite("fieldBits", &PublicParameters::fieldBits);

    m.attr("HAS_EC") =
#ifdef VPSS_WITH_EC
        true;
#else
        false;
#endif

#ifdef VPSS_WITH_EC
    m.def("ec_setup", [](const std::string& curve) {
        int nid;
        if (curve == "P-256" || curve == "secp256r1" || curve == "prime256v1")
            nid = NID_X9_62_prime256v1;
        else if (curve == "P-384" || curve == "secp384r1")
            nid = NID_secp384r1;
        else if (curve == "P-521" || curve == "secp521r1")
            nid = NID_secp521r1;
        else
            throw std::invalid_argument("unsupported curve: " + curve);

        const EC_GROUP* grp = ecbackend::group(nid);
        BN_CTX* c = ecbackend::ctx();

        BIGNUM* pf = BN_new();
        BIGNUM* ca = BN_new();
        BIGNUM* cb = BN_new();
        if (!EC_GROUP_get_curve(grp, pf, ca, cb, c)) {
            BN_free(pf); BN_free(ca); BN_free(cb);
            throw std::runtime_error("EC_GROUP_get_curve failed");
        }
        unsigned fieldBits = static_cast<unsigned>(((BN_num_bits(pf) + 7) / 8) * 8);
        BigInt fieldPrime = ecbackend::fromBn(pf);
        BN_free(ca); BN_free(cb);

        BigInt order = ecbackend::fromBn(EC_GROUP_get0_order(grp));
        BigInt G = ecbackend::encode(nid, fieldBits, EC_GROUP_get0_generator(grp));

        // h is derived by hashing a fixed string, so nobody -- including whoever generates
        // the parameters -- knows log_g(h). Pedersen binding rests on that.
        size_t fieldBytes = static_cast<size_t>((BN_num_bits(pf) + 7) / 8);
        BigInt H(0);
        bool found = false;
        for (int i = 0; i < 4096 && !found; ++i) {
            std::vector<unsigned char> stream;
            for (int blk = 0; stream.size() < fieldBytes; ++blk) {
                std::string msg = "RVFL-pedersen-h|" + curve + "|" + std::to_string(i) +
                                  "|" + std::to_string(blk);
                unsigned char digest[SHA256_DIGEST_LENGTH];
                SHA256(reinterpret_cast<const unsigned char*>(msg.data()), msg.size(), digest);
                stream.insert(stream.end(), digest, digest + SHA256_DIGEST_LENGTH);
            }
            BigInt x(0);
            mpz_import(x.get_mpz_t(), fieldBytes, 1, 1, 1, 0, stream.data());
            x = x % fieldPrime;
            BIGNUM* xb = ecbackend::toBn(x);
            ecbackend::Point pt(grp);
            int ok = EC_POINT_set_compressed_coordinates(grp, pt.p, xb, 0, c);
            BN_free(xb);
            if (ok && EC_POINT_is_on_curve(grp, pt.p, c) && !EC_POINT_is_at_infinity(grp, pt.p)) {
                H = ecbackend::encode(nid, fieldBits, pt.p);
                found = true;
            }
        }
        BN_free(pf);
        if (!found) throw std::runtime_error("failed to derive h");

        py::dict out;
        out["nid"] = nid;
        out["field_bits"] = fieldBits;
        out["p"] = fieldPrime;
        out["q"] = order;
        out["g"] = G;
        out["h"] = H;
        return out;
    },
    py::arg("curve"),
    "Curve parameters for the EC commitment backend. curve is P-256, P-384 or P-521.");
#endif


    py::class_<VPSSConfig>(m, "VPSSConfig")
        .def(py::init<>())
        .def_readwrite("t", &VPSSConfig::t)
        .def_readwrite("k", &VPSSConfig::k)
        .def_readwrite("N", &VPSSConfig::N)
        .def_readwrite("deterministicRand", &VPSSConfig::deterministicRand)
        .def_readwrite("padType", &VPSSConfig::padType)
        .def_readwrite("secretIndices", &VPSSConfig::secretIndices)
        .def_readwrite("shareIndices", &VPSSConfig::shareIndices);

    // -------------------------------------------------------------------------
    // MathHelper Matrix Operations Binding
    // -------------------------------------------------------------------------
    
    // create Vandermonde Matrix 
    m.def("create_vandermonde_matrix", &MathHelper::createVandermondeMatrix,
          "Creates a Vandermonde matrix: N x (degree+1)",
          py::arg("x_coords"), py::arg("degree"), py::arg("mod"));

    // Matrix Transpose (A^T)
    m.def("transpose_matrix", &MathHelper::transposeMatrix,
          "Transposes a matrix",
          py::arg("matrix"));

    // Matrix Inversion (A^-1)
    m.def("invert_matrix", &MathHelper::invertMatrix,
          "Computes modular inverse of a square matrix",
          py::arg("matrix"), py::arg("mod"));

    // Matrix Multiplication (A * B)
    m.def("multiply_matrix", 
          static_cast<Matrix (*)(const Matrix&, const Matrix&, const BigInt&)>(
              &MathHelper::multiplyMatrix),
          "Multiplies two matrices modulo q",
          py::arg("A"), py::arg("B"), py::arg("mod"));
    
    // Matrix * Vector (Result: N x 1 Matrix)
    m.def("multiply_matrix_vec", 
          [](const Matrix& A, const std::vector<BigInt>& vec, const BigInt& mod) {
              Matrix mat_res = MathHelper::multiplyMatrix(A, vec, mod);
              
              std::vector<BigInt> flat_res;
              flat_res.reserve(mat_res.size());
              for (const auto& row : mat_res) {
                  if (!row.empty()) {
                      flat_res.push_back(row[0]);
                  } else {
                      flat_res.push_back(BigInt(0));
                  }
              }
              return flat_res;
          },
          "Multiplies Matrix by Column Vector and returns flattened Vector",
          py::arg("A"), py::arg("vec"), py::arg("mod"));

    // -------------------------------------------------------------------------
    // Helper: Single Point Verification (Consistency Check)
    // -------------------------------------------------------------------------
    m.def("verify_Pedersen", 
          [](const PublicParameters& params, 
             const BigInt& x,
             const BigInt& s,
             const BigInt& r,
             const std::vector<BigInt>& commitment) {
            Share<VerificationMode::Pedersen> tempShare;
            tempShare.x = x;
            tempShare.s = s;
            tempShare.r = r;
            return UnifiedVPSS<VerificationMode::Pedersen>::verifyShare(params, tempShare, commitment);
        },
        "Verifies consistency: g^s == Prod(C_k * x^k) mod p",
        py::arg("params"), py::arg("x"), py::arg("s"), py::arg("r"), py::arg("commitment"));

    // -------------------------------------------------------------------------
    // BatchShare Definitions
    // -------------------------------------------------------------------------
    py::bind_vector<std::vector<BatchSharePedersen>>(m, "BatchSharePedersenVector")
        .def(py::pickle(
            [](const std::vector<BatchSharePedersen>& self) { 
                Serializer s; s.write_vector_share_pedersen(self); return s.to_bytes(); 
            },
            [](py::bytes state) { 
                Deserializer d(state); return d.read_vector_share_pedersen(); 
            }
        ));
    py::bind_vector<std::vector<BatchShareFeldman>>(m, "BatchShareFeldmanVector")
        .def(py::pickle(
            [](const std::vector<BatchShareFeldman>& self) { 
                Serializer s; s.write_vector_share_feldman(self); return s.to_bytes(); 
            },
            [](py::bytes state) { 
                Deserializer d(state); return d.read_vector_share_feldman(); 
            }
        ));
    py::bind_vector<std::vector<BatchShareNone>>(m, "BatchShareNoneVector")
        .def(py::pickle(
            [](const std::vector<BatchShareNone>& self) { 
                Serializer s; s.write_vector_share_none(self); return s.to_bytes(); 
            },
            [](py::bytes state) { 
                Deserializer d(state); return d.read_vector_share_none(); 
            }
        ));

    py::class_<BatchSharePedersen>(m, "BatchSharePedersen")
        .def(py::init<>())
        .def_readwrite("x", &BatchSharePedersen::x)
        .def_readwrite("s_values", &BatchSharePedersen::s_values)
        .def_readwrite("r_values", &BatchSharePedersen::r_values)
        .def(py::pickle(
            [](const BatchSharePedersen& self) {
                Serializer s; s.write_share_pedersen(self); return s.to_bytes();
            },
            [](py::bytes state) {
                BatchSharePedersen share;
                Deserializer d(state); d.read_share_pedersen(share);
                return share;
            }
        ));

    py::class_<BatchShareFeldman>(m, "BatchShareFeldman")
        .def(py::init<>())
        .def_readwrite("x", &BatchShareFeldman::x)
        .def_readwrite("s_values", &BatchShareFeldman::s_values)
        .def(py::pickle(
            [](const BatchShareFeldman& self) {
                Serializer s; s.write_share_feldman(self); return s.to_bytes();
            },
            [](py::bytes state) {
                BatchShareFeldman share;
                Deserializer d(state); d.read_share_feldman(share);
                return share;
            }
        ));

    py::class_<BatchShareNone>(m, "BatchShareNone")
        .def(py::init<>())
        .def_readwrite("x", &BatchShareNone::x)
        .def_readwrite("s_values", &BatchShareNone::s_values)
        .def(py::pickle(
            [](const BatchShareNone& self) { Serializer s; s.write_share_none(self); return s.to_bytes(); },
            [](py::bytes state) { BatchShareNone share; Deserializer d(state); d.read_share_none(share); return share; }
        ));

    // -------------------------------------------------------------------------
    // BatchVPSS Bindings
    // -------------------------------------------------------------------------
    #define BIND_BATCH_VPSS(ModeEnum, ClassName) \
    py::class_<PyBatchVPSS<ModeEnum>>(m, ClassName) \
        .def(py::init<>()) \
        .def_static("distribute", &PyBatchVPSS<ModeEnum>::distribute, \
            py::arg("params"), py::arg("config"), py::arg("secrets"), \
            py::arg("thread_count") = 1, py::arg("return_blinds") = false) \
        .def_static("verify_batch_share", &PyBatchVPSS<ModeEnum>::verify_batch_share, \
             py::arg("params"), py::arg("share"), py::arg("commitments"), py::arg("thread_count") = 1) \
        .def_static("reconstruct", &PyBatchVPSS<ModeEnum>::reconstruct, \
             py::arg("params"), py::arg("config"), py::arg("batchShares"), \
             py::arg("useErrorCorrection") = false, py::arg("thread_count") = 1)

    BIND_BATCH_VPSS(VerificationMode::Pedersen, "PedersenVPSS");
    BIND_BATCH_VPSS(VerificationMode::Feldman, "FeldmanVPSS");
    BIND_BATCH_VPSS(VerificationMode::None, "NoneVPSS");
}