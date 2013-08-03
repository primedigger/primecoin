#define MAX_CANDIDATES 2400

#define DIGITS_CAPACITY 16

/** @brief struct used to represent multiple precision integers (Z). */
typedef struct {  
  uint32_t capacity;
  uint32_t digits[DIGITS_CAPACITY];
  uint32_t sign;
} mpz_cuda_t;

struct cudaCandidate {
    #ifdef CUDA_DEBUG
    char strChainOrigin[256];
    #endif
    //char strPrimeChainMultiplier[512];
    mpz_cuda_t chainOrigin;
    unsigned int blocknBits;
    unsigned int nChainLengthCunningham1;
    unsigned int nChainLengthCunningham2;
    unsigned int nChainLengthBiTwin;
};

void runCandidateSearchKernel(cudaCandidate *candidates, char *result, unsigned int num_candidates);
