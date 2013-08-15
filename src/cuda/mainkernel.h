#define MAX_CANDIDATES 9600

#define DIGITS_CAPACITY 32

/** @brief struct used to represent multiple precision integers (Z). */
typedef struct {  
  uint32_t capacity;
  uint32_t digits[DIGITS_CAPACITY];
  uint32_t sign;
} mpz_cuda_t;

typedef struct {  
  uint32_t capacity;
  uint32_t digits[DIGITS_CAPACITY*2];
  uint32_t sign;
} mpz_cuda_t_wide;

struct cudaCandidate {
    #ifdef CUDA_DEBUG
    //char strChainOrigin[256];
    #endif
    //char strPrimeChainMultiplier[512];
    mpz_cuda_t chainOrigin;
    unsigned int blocknBits;
    unsigned int nChainLengthCunningham1;
    unsigned int nChainLengthCunningham2;
    unsigned int nChainLengthBiTwin;
};

void runCandidateSearchKernel(cudaCandidate *candidates, char *result, unsigned int num_candidates);
