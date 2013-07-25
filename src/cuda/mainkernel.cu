//#define CUDA_DEBUG


#include <stdio.h>

typedef unsigned int uint32_t;


#include "mainkernel.h"
#include "mpz.h"    // multiple precision cuda code
#include "cuda_string.h"


//__device__ mpz_cuda_t mpzTemp;

#define mpz_clear mpz_destroy
#define mpz_cmp mpz_compare
#define mpz_mul mpz_mult
#define mpz_powm mpz_powmod

//copied constants from prime.h

static const unsigned int nFractionalBits = 24;
static const unsigned int TARGET_FRACTIONAL_MASK = (1u<<nFractionalBits) - 1;
static const unsigned int TARGET_LENGTH_MASK = ~TARGET_FRACTIONAL_MASK;
//static const uint64 nFractionalDifficultyMax = (1llu << (nFractionalBits + 32));
//static const uint64 nFractionalDifficultyMin = (1llu << 32);
//static const uint64 nFractionalDifficultyThreshold = (1llu << (8 + 32));
//static const unsigned int nWorkTransitionRatio = 32;

//end copy

//mpz_div(mpz_cuda_t *q, mpz_cuda_t *r, mpz_cuda_t *n, mpz_cuda_t *d)


//extra mpz_functions (quick and dirty...)
__device__ inline void mpz_tdiv_q(mpz_cuda_t *ROP, mpz_cuda_t *OP1, mpz_cuda_t *OP2)
{
    mpz_cuda_t mpzTemp;
    mpz_init(&mpzTemp);
    mpz_div(ROP,&mpzTemp,OP1,OP2);
    mpz_destroy(&mpzTemp);
}

__device__ inline void mpz_tdiv_r(mpz_cuda_t *ROP, mpz_cuda_t *OP1, mpz_cuda_t *OP2)
{
    mpz_cuda_t mpzTemp;
    mpz_init(&mpzTemp);
    mpz_div(&mpzTemp,ROP,OP1,OP2);
    mpz_destroy(&mpzTemp);
}

__device__ inline unsigned int mpz_get_ui(mpz_cuda_t *OP)
{
    return OP->digits[0];
}

//Set product to multiplicator times 2 raised to exponent_of_2. This operation can also be defined as a left shift, exponent_of_2 steps.
__device__ inline void mpz_mul_2exp (mpz_cuda_t *product, mpz_cuda_t *multiplicator, unsigned long int exponent_of_2)
{
    mpz_cuda_t mpzTemp;
    mpz_init(&mpzTemp);
    mpz_set_ui(&mpzTemp,2);
    unsigned int limit = exponent_of_2;
    //well this is ugly
    for(unsigned int i=0; i < limit; i++)
    	mpz_bit_lshift(&mpzTemp);

    mpz_mul(product,multiplicator,&mpzTemp);
    mpz_destroy(&mpzTemp);
}


//end extra mpz

__device__ bool devTargetSetLength(unsigned int nLength, unsigned int& nBits)
{
    if (nLength >= 0xff)
    {
        printf("[CUDA] error TargetSetLength() : invalid length=%u\n", nLength);
	return false;
    }
    nBits &= TARGET_FRACTIONAL_MASK;
    nBits |= (nLength << nFractionalBits);
    return true;
}

__device__ unsigned int devTargetGetLength(unsigned int nBits)
{
    return ((nBits & TARGET_LENGTH_MASK) >> nFractionalBits);
}

__device__ unsigned int devTargetFromInt(unsigned int nLength)
{
    return (nLength << nFractionalBits);
}

__device__ void devTargetIncrementLength(unsigned int& nBits)
{
    nBits += (1 << nFractionalBits);
}

// Check Fermat probable primality test (2-PRP): 2 ** (n-1) = 1 (mod n)
// true: n is probable prime
// false: n is composite; set fractional length in the nLength output
__device__ bool devFermatProbablePrimalityTest(mpz_cuda_t &mpzN, unsigned int& nLength)
{
    mpz_cuda_t mpzOne;
    mpz_cuda_t mpzTwo;
    //mpz_cuda_t mpzEight;

    //TODO: generate constants in a different kernel
    mpz_init(&mpzOne);
    mpz_set_ui(&mpzOne,1);	

    mpz_init(&mpzTwo);
    mpz_set_ui(&mpzTwo,2);

    //mpz_init(&mpzEight);
    //mpz_set_ui(&mpzEight,8);

    // Faster GMP version
    
    //mpz_cuda_t mpzN;
    mpz_cuda_t mpzE;
    mpz_cuda_t mpzR;
    
    //mpz_init_set(mpzN, n.get_mpz_cuda_t());

    //e = n -1

    mpz_init(&mpzE);
    mpz_sub(&mpzE, &mpzN, &mpzOne);
    mpz_init(&mpzR);


    //BN_mod_exp(&r, &a, &e, &n);
	// r = 2^(n-1) & n
    mpz_powm(&mpzR, &mpzTwo, &mpzE, &mpzN);

    mpz_destroy(&mpzOne);
    mpz_destroy(&mpzTwo);

    if (mpz_cmp(&mpzR, &mpzOne) == 0)
    {
        mpz_clear(&mpzN);
        mpz_clear(&mpzE);
        mpz_clear(&mpzR);
        
        printf("[CUDA] Fermat test true\n");
        return true;
    }
    // Failed Fermat test, calculate fractional length
    mpz_sub(&mpzE, &mpzN, &mpzR);
    mpz_mul_2exp(&mpzR, &mpzE, nFractionalBits);
    mpz_tdiv_q(&mpzE, &mpzR, &mpzN);

    unsigned int nFractionalLength = mpz_get_ui(&mpzE);
    mpz_clear(&mpzN);
    mpz_clear(&mpzE);
    mpz_clear(&mpzR);

    if (nFractionalLength >= (1 << nFractionalBits))
    {
	printf("[CUDA] Error FermatProbablePrimalityTest() : fractional assert : nFractionalLength:%i nFractionalBits:%i\n", nFractionalLength, nFractionalBits);
        return false;
    }

    nLength = (nLength & TARGET_LENGTH_MASK) | nFractionalLength;
    return false;
}

//this version prints results for thread 0
__device__ bool devFermatProbablePrimalityTestWithPrint(mpz_cuda_t &mpzN, unsigned int& nLength, unsigned int index)
{
    bool prime = false;

    mpz_cuda_t mpzOne;
    mpz_cuda_t mpzTwo;
    //mpz_cuda_t mpzEight;

    //TODO: generate constants in a different kernel
    mpz_init(&mpzOne);
    mpz_set_ui(&mpzOne,1);	

    mpz_init(&mpzTwo);
    mpz_set_ui(&mpzTwo,2);

    //mpz_init(&mpzEight);
    //mpz_set_ui(&mpzEight,8);

    // Faster GMP version
    
    //mpz_cuda_t mpzN;
    mpz_cuda_t mpzE;
    mpz_cuda_t mpzR;
    
    //mpz_init_set(mpzN, n.get_mpz_cuda_t());

    //e = n -1

    mpz_init(&mpzE);
    mpz_sub(&mpzE, &mpzN, &mpzOne);

#ifdef CUDA_DEBUG
    if(index == 0)
    {
	printf("[0] N is: ");
	mpz_print(&mpzN);
	printf("[0] E is: ");
	mpz_print(&mpzE);
    }
#endif

    mpz_init(&mpzR);

    //BN_mod_exp(&r, &a, &e, &n);
    mpz_powm(&mpzR, &mpzTwo, &mpzE, &mpzN);

#ifdef CUDA_DEBUG
    if(index == 0)
    {
	printf("[0] R is: ");
	mpz_print(&mpzR);
    }
#endif

    mpz_destroy(&mpzOne);
    mpz_destroy(&mpzTwo);

    if (mpz_cmp(&mpzR, &mpzOne) == 0)
    {
	prime = true;  
	#ifdef CUDA_DEBUG
	if(index == 0)      
        	printf("[0] Fermat test true\n");
	#endif
    }

    mpz_clear(&mpzN);
    mpz_clear(&mpzE);
    mpz_clear(&mpzR);

    return prime;
    // Failed Fermat test, calculate fractional length
    /*mpz_sub(&mpzE, &mpzN, &mpzR);
    mpz_mul_2exp(&mpzR, &mpzE, nFractionalBits);
    mpz_tdiv_q(&mpzE, &mpzR, &mpzN);

    unsigned int nFractionalLength = mpz_get_ui(&mpzE);
    mpz_clear(&mpzN);
    mpz_clear(&mpzE);
    mpz_clear(&mpzR);

    if (nFractionalLength >= (1 << nFractionalBits))
    {
	if(index==0)
		printf("[CUDA] Error FermatProbablePrimalityTest() : fractional assert : nFractionalLength:%i nFractionalBits:%i\n", nFractionalLength, nFractionalBits);
        return false;
    }

    nLength = (nLength & TARGET_LENGTH_MASK) | nFractionalLength;
    return false;*/
}

// Test probable primality of n = 2p +/- 1 based on Euler, Lagrange and Lifchitz
// fSophieGermain:
//   true:  n = 2p+1, p prime, aka Cunningham Chain of first kind
//   false: n = 2p-1, p prime, aka Cunningham Chain of second kind
// Return values
//   true: n is probable prime
//   false: n is composite; set fractional length in the nLength output
__device__ bool devEulerLagrangeLifchitzPrimalityTest(mpz_cuda_t &mpzN, bool fSophieGermain, unsigned int& nLength)
{

    mpz_cuda_t mpzOne;
    mpz_cuda_t mpzTwo;
    //mpz_cuda_t mpzEight;

    //TODO: generate constants in a different kernel
    mpz_init(&mpzOne);
    mpz_set_ui(&mpzOne,1);	

    mpz_init(&mpzTwo);
    mpz_set_ui(&mpzTwo,2);

    //mpz_init(&mpzEight);
    //mpz_set_ui(&mpzEight,8);

    // Faster GMP version
    //mpz_cuda_t mpzN;
    mpz_cuda_t mpzE;
    mpz_cuda_t mpzR;
    mpz_cuda_t temp;

    mpz_init(&temp);    

    mpz_init(&mpzE);
    mpz_sub(&mpzE, &mpzN, &mpzOne);
 
    //mpz_set(&temp,&mpzE);

   //e = (n - 1) >> 1;
    //from hp4: mpz_tdiv_q_2exp(&mpzE, &mpzE, 1);
    mpz_tdiv_q(&temp,&mpzE,&mpzTwo);
    mpz_set(&mpzE,&temp);

    mpz_destroy(&temp);

    mpz_init(&mpzR);
    mpz_powm(&mpzR, &mpzTwo, &mpzE, &mpzN);
   
    //nMod8 = n % 8; 
    //mpz_cuda_t mpzNMod8;
    //mpz_init(&mpzNMod8);
    //mpz_tdiv_r(&mpzNMod8,&mpzN, &mpzEight);
    unsigned int nMod8 = mpz_get_ui(&mpzN) % 8;    
    //mpz_destroy(&mpzNMod8);

    bool fPassedTest = false;
    if (fSophieGermain && (nMod8 == 7)) // Euler & Lagrange
        fPassedTest = !mpz_cmp(&mpzR, &mpzOne);
    else if (fSophieGermain && (nMod8 == 3)) // Lifchitz
    {
        mpz_cuda_t mpzRplusOne;
        mpz_init(&mpzRplusOne);
        mpz_add(&mpzRplusOne, &mpzR, &mpzOne);
        fPassedTest = !mpz_cmp(&mpzRplusOne, &mpzN);
        mpz_clear(&mpzRplusOne);
    }
    else if ((!fSophieGermain) && (nMod8 == 5)) // Lifchitz
    {
        mpz_cuda_t mpzRplusOne;
        mpz_init(&mpzRplusOne);
        mpz_add(&mpzRplusOne, &mpzR, &mpzOne);
        fPassedTest = !mpz_cmp(&mpzRplusOne, &mpzN);
        mpz_clear(&mpzRplusOne);
    }
    else if ((!fSophieGermain) && (nMod8 == 1)) // LifChitz
    {
        fPassedTest = !mpz_cmp(&mpzR, &mpzOne);
    }
    else
    {
        mpz_clear(&mpzN);
        mpz_clear(&mpzE);
        mpz_clear(&mpzR);
        mpz_destroy(&mpzOne);
        mpz_destroy(&mpzTwo);
        printf("[CUDA] Error in EulerLagrangeLifchitzPrimalityTest() : invalid n %% 8 = %d, %s", nMod8, (fSophieGermain? "first kind" : "second kind"));
        return false;
    }
    
    if (fPassedTest)
    {
        mpz_clear(&mpzN);
        mpz_clear(&mpzE);
        mpz_clear(&mpzR);
	mpz_destroy(&mpzOne);
        mpz_destroy(&mpzTwo);
        return true;
    }
    
    // Failed test, calculate fractional length
    //TODO: RCOPY
    mpz_mul(&mpzE, &mpzR, &mpzR);
    mpz_tdiv_r(&mpzR, &mpzE, &mpzN); // derive Fermat test remainder

    mpz_sub(&mpzE, &mpzN, &mpzR);
    mpz_mul_2exp(&mpzR, &mpzE, nFractionalBits);
    mpz_tdiv_q(&mpzE, &mpzR, &mpzN);

    //Todo: implement mpz_get_ui
    unsigned int nFractionalLength = mpz_get_ui(&mpzE);
    mpz_clear(&mpzN);
    mpz_clear(&mpzE);
    mpz_clear(&mpzR);
    mpz_destroy(&mpzOne);
    mpz_destroy(&mpzTwo);
    
    if (nFractionalLength >= (1 << nFractionalBits))
    {
        printf("[CUDA] error EulerLagrangeLifchitzPrimalityTest() : fractional assert");
        return false;
    }
    nLength = (nLength & TARGET_LENGTH_MASK) | nFractionalLength;
    return false;
}



// Test Probable Cunningham Chain for: n
// fSophieGermain:
//   true - Test for Cunningham Chain of first kind (n, 2n+1, 4n+3, ...)
//   false - Test for Cunningham Chain of second kind (n, 2n-1, 4n-3, ...)
// Return value:
//   true - Probable Cunningham Chain found (length at least 2)
//   false - Not Cunningham Chain
__device__ bool devProbableCunninghamChainTest(mpz_cuda_t &n, bool fSophieGermain, bool fFermatTest, unsigned int& nProbableChainLength)
{
    nProbableChainLength = 0;
    //mpz_class N = n;

    mpz_cuda_t N;
    mpz_init(&N);

    mpz_cuda_t N_copy;
    mpz_init(&N_copy);

    mpz_set(&N,&n);    

    // Fermat test for n first
    if (!devFermatProbablePrimalityTest(N, nProbableChainLength))
        return false;

    printf("[CUDA ] N is prime!\n");

    // Euler-Lagrange-Lifchitz test for the following numbers in chain
    while (true)
    {
        devTargetIncrementLength(nProbableChainLength);
	//N = N + N or N *=2
	mpz_set(&N_copy,&N);  
        mpz_mult_u(&N,&N_copy,2);
	// N+ = (fSophieGermain? 1 : (-1))
	mpz_addeq_i(&N,(fSophieGermain? 1 : (-1)));
        if (fFermatTest)
        {
            if (!devFermatProbablePrimalityTest(N, nProbableChainLength))
                break;
        }
        else
        {
            if (!devEulerLagrangeLifchitzPrimalityTest(N, fSophieGermain, nProbableChainLength))
                break;
        }
    }

    mpz_destroy(&N);
    mpz_destroy(&N_copy);

    return (devTargetGetLength(nProbableChainLength) >= 2);
}

// Test probable prime chain for: nOrigin
// Return value:
//   true - Probable prime chain found (one of nChainLength meeting target)
//   false - prime chain too short (none of nChainLength meeting target)
__device__ bool devProbablePrimeChainTest(mpz_cuda_t &mpzPrimeChainOrigin, unsigned int nBits, bool fFermatTest, unsigned int& nChainLengthCunningham1, unsigned int& nChainLengthCunningham2, unsigned int& nChainLengthBiTwin)
{
    mpz_cuda_t mpzOne;
    mpz_init(&mpzOne);
    mpz_set_ui(&mpzOne,1);

    nChainLengthCunningham1 = 0;
    nChainLengthCunningham2 = 0;
    nChainLengthBiTwin = 0;

    mpz_cuda_t mpzPrimeChainOriginMinusOne;
    mpz_cuda_t mpzPrimeChainOriginPlusOne;

    mpz_init(&mpzPrimeChainOriginMinusOne);
    mpz_init(&mpzPrimeChainOriginPlusOne);

    mpz_add(&mpzPrimeChainOriginPlusOne,&mpzPrimeChainOrigin,&mpzOne);
    mpz_sub(&mpzPrimeChainOriginMinusOne,&mpzPrimeChainOrigin,&mpzOne);

    // Test for Cunningham Chain of first kind
    devProbableCunninghamChainTest(mpzPrimeChainOriginMinusOne, true, fFermatTest, nChainLengthCunningham1);
    // Test for Cunningham Chain of second kind
    devProbableCunninghamChainTest(mpzPrimeChainOriginPlusOne, false, fFermatTest, nChainLengthCunningham2);
    // Figure out BiTwin Chain length
    // BiTwin Chain allows a single prime at the end for odd length chain
    nChainLengthBiTwin =
        (devTargetGetLength(nChainLengthCunningham1) > devTargetGetLength(nChainLengthCunningham2))?
            (nChainLengthCunningham2 + devTargetFromInt(devTargetGetLength(nChainLengthCunningham2)+1)) :
            (nChainLengthCunningham1 + devTargetFromInt(devTargetGetLength(nChainLengthCunningham1)));

    mpz_destroy(&mpzPrimeChainOriginMinusOne);
    mpz_destroy(&mpzPrimeChainOriginPlusOne);
    mpz_destroy(&mpzOne);

    return (nChainLengthCunningham1 >= nBits || nChainLengthCunningham2 >= nBits || nChainLengthBiTwin >= nBits);
}

__global__ void runPrimeCandidateSearch(cudaCandidate *candidates, char *result, unsigned int num_candidates)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	//even index threads do -1 fermat test for n=index/2
	//odd index threads do +1 fermat test for n=index/2

	//check bounds
	if (index < 2*num_candidates)
	{
	    mpz_cuda_t mpzN;
	    mpz_init(&mpzN);

		#ifdef CUDA_DEBUG
		if(index==0)
		{
			printf("[0] start! \n");
			printf("sizeof(struct) = %i\n",sizeof(cudaCandidate));		
		}
		#endif

		cudaCandidate candidate = candidates[index/2];

		#ifdef CUDA_DEBUG
		if(index==0)
		{
			printf("mpz_print:");
			mpz_print( &candidate.chainOrigin);
			
			//for (int i=0; i < candidate.chainOrigin.capacity; i++)
			//	printf("%x\n",candidate.chainOrigin.digits[i]);

			printf("[0] string candidate is %s\n",candidate.strChainOrigin);
		}
		#endif

		mpz_set(&mpzN,&candidate.chainOrigin);
		mpz_addeq_i(&mpzN, index % 2 == 0 ? -1 : 1);

		unsigned int nLength=0;

		char testresult = 0x00;

		//test for chain of length two
		if(devFermatProbablePrimalityTestWithPrint(mpzN, nLength, index))
		{
			testresult = 0x01;
		}
		
		result[index/2] = testresult;
		
#ifdef CUDA_DEBUG
		if(index==0)
			printf("[0] after fermat test\n");
#endif

		mpz_destroy(&mpzN);
	}
}


void runCandidateSearchKernel(cudaCandidate *candidates, char *result, unsigned int num_candidates)
{
	//TODO: make gridsize dynamic
	runPrimeCandidateSearch<<< 25 , 192>>>(candidates, result, num_candidates);

}
