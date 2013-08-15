// Copyright (c) 2013 Primecoin developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef PRIMECOIN_PRIME_H
#define PRIMECOIN_PRIME_H

#include "main.h"

#include <gmp.h>
#include <gmpxx.h>
#include <bitset>

/**********************/
/* PRIMECOIN PROTOCOL */
/**********************/

static const unsigned int nMaxRoundSievePercentage = 100;
static const unsigned int nDefaultRoundSievePercentage = 70;
static const unsigned int nDefaultRoundSievePercentageTestnet = 30;
static const unsigned int nMinRoundSievePercentage = 1;
extern unsigned int nRoundSievePercentage;
static const unsigned int nMaxSievePercentage = 100;
static const unsigned int nDefaultSievePercentage = 10;
static const unsigned int nMinSievePercentage = 1;
extern unsigned int nSievePercentage;
static const unsigned int nMaxSieveSize = 40000000u;
static const unsigned int nDefaultSieveSize = 4000000u;
static const unsigned int nMinSieveSize = 100000u;
extern unsigned int nSieveSize;
static const uint256 hashBlockHeaderLimit = (uint256(1) << 255);
static const CBigNum bnOne = 1;
static const CBigNum bnPrimeMax = (bnOne << 2000) - 1;
static const CBigNum bnPrimeMin = (bnOne << 255);
static const mpz_class mpzOne = 1;
static const mpz_class mpzTwo = 2;
static const mpz_class mpzPrimeMax = (mpzOne << 2000) - 1;
static const mpz_class mpzPrimeMin = (mpzOne << 255);

// Estimate how many 5-chains are found per hour
static const unsigned int nStatsChainLength = 5;

extern unsigned int nTargetInitialLength;
extern unsigned int nTargetMinLength;

// Generate small prime table
void GeneratePrimeTable();
// Get next prime number of p
bool PrimeTableGetNextPrime(unsigned int& p);
// Get previous prime number of p
bool PrimeTableGetPreviousPrime(unsigned int& p);

// Compute primorial number p#
void Primorial(unsigned int p, mpz_class& mpzPrimorial);
// Compute Primorial number p#
// Fast 32-bit version assuming that p <= 23
unsigned int PrimorialFast(unsigned int p);
// Compute the first primorial number greater than or equal to bn
void PrimorialAt(mpz_class& bn, mpz_class& mpzPrimorial);

// Test probable prime chain for: bnPrimeChainOrigin
// fFermatTest
//   true - Use only Fermat tests
//   false - Use Fermat-Euler-Lagrange-Lifchitz tests
// Return value:
//   true - Probable prime chain found (one of nChainLength meeting target)
//   false - prime chain too short (none of nChainLength meeting target)
bool ProbablePrimeChainTest(const CBigNum& bnPrimeChainOrigin, unsigned int nBits, bool fFermatTest, unsigned int& nChainLengthCunningham1, unsigned int& nChainLengthCunningham2, unsigned int& nChainLengthBiTwin);

static const unsigned int nFractionalBits = 24;
static const unsigned int TARGET_FRACTIONAL_MASK = (1u<<nFractionalBits) - 1;
static const unsigned int TARGET_LENGTH_MASK = ~TARGET_FRACTIONAL_MASK;
static const uint64 nFractionalDifficultyMax = (1llu << (nFractionalBits + 32));
static const uint64 nFractionalDifficultyMin = (1llu << 32);
static const uint64 nFractionalDifficultyThreshold = (1llu << (8 + 32));
static const unsigned int nWorkTransitionRatio = 32;
unsigned int TargetGetLimit();
unsigned int TargetGetInitial();
unsigned int TargetGetLength(unsigned int nBits);
bool TargetSetLength(unsigned int nLength, unsigned int& nBits);
unsigned int TargetGetFractional(unsigned int nBits);
uint64 TargetGetFractionalDifficulty(unsigned int nBits);
bool TargetSetFractionalDifficulty(uint64 nFractionalDifficulty, unsigned int& nBits);
std::string TargetToString(unsigned int nBits);
unsigned int TargetFromInt(unsigned int nLength);
bool TargetGetMint(unsigned int nBits, uint64& nMint);
bool TargetGetNext(unsigned int nBits, int64 nInterval, int64 nTargetSpacing, int64 nActualSpacing, unsigned int& nBitsNext);

// Check prime proof-of-work
enum // prime chain type
{
    PRIME_CHAIN_CUNNINGHAM1 = 1u,
    PRIME_CHAIN_CUNNINGHAM2 = 2u,
    PRIME_CHAIN_BI_TWIN     = 3u,
};
bool CheckPrimeProofOfWork(uint256 hashBlockHeader, unsigned int nBits, const CBigNum& bnPrimeChainMultiplier, unsigned int& nChainType, unsigned int& nChainLength);

// prime target difficulty value for visualization
double GetPrimeDifficulty(unsigned int nBits);
// Estimate work transition target to longer prime chain
unsigned int EstimateWorkTransition(unsigned int nPrevWorkTransition, unsigned int nBits, unsigned int nChainLength);
// prime chain type and length value
std::string GetPrimeChainName(unsigned int nChainType, unsigned int nChainLength);
// primorial form of prime chain origin
std::string GetPrimeOriginPrimorialForm(CBigNum& bnPrimeChainOrigin);


/********************/
/* PRIMECOIN MINING */
/********************/

// Mine probable prime chain of form: n = h * p# +/- 1
bool MineProbablePrimeChain(CBlock& block, mpz_class& mpzFixedMultiplier, bool& fNewBlock, unsigned int& nTriedMultiplier, unsigned int& nProbableChainLength, unsigned int& nTests, unsigned int& nPrimesHit, unsigned int& nChainsHit, mpz_class& mpzHash, unsigned int nPrimorialMultiplier, int64& nSieveGenTime, CBlockIndex* pindexPrev);

// Estimate the probability of primality for a number in a candidate chain
double EstimateCandidatePrimeProbability(unsigned int nPrimorialMultiplier);

#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__x86_64__) || defined(_M_X64)
#define USE_ROTATE
#endif

// Sieve of Eratosthenes for proof-of-work mining
class CSieveOfEratosthenes
{
    unsigned int nSieveSize; // size of the sieve
    unsigned int nBits; // target of the prime chain to search for
    mpz_class mpzHash; // hash of the block header
    mpz_class mpzFixedMultiplier; // fixed round multiplier

    // final set of candidates for probable primality checking
    unsigned long *vfCandidates;
    unsigned long *vfCandidateBiTwin;
    unsigned long *vfCandidateCunningham1;
    
    static const unsigned int nWordBits = 8 * sizeof(unsigned long);
    unsigned int nCandidatesWords;
    unsigned int nCandidatesBytes;

    unsigned int nPrimeSeq; // prime sequence number currently being processed
    unsigned int nCandidateCount; // cached total count of candidates
    unsigned int nCandidateMultiplier; // current candidate for power test
    
    unsigned int nChainLength;
    unsigned int nHalfChainLength;
    unsigned int nPrimes;
    
    CBlockIndex* pindexPrev;
    
    unsigned int GetWordNum(unsigned int nBitNum) {
        return nBitNum / nWordBits;
    }
    
    unsigned long GetBitMask(unsigned int nBitNum) {
        return 1UL << (nBitNum % nWordBits);
    }
    
    void AddMultiplier(unsigned int *vMultipliers, const unsigned int nPrimeSeq, const unsigned int nSolvedMultiplier);

    void ProcessMultiplier(unsigned long *vfComposites, const unsigned int nMinMultiplier, const unsigned int nMaxMultiplier, const std::vector<unsigned int>& vPrimes, unsigned int *vMultipliers)
    {
        // Wipe the part of the array first
        memset(vfComposites + GetWordNum(nMinMultiplier), 0, (nMaxMultiplier - nMinMultiplier + nWordBits - 1) / nWordBits * sizeof(unsigned long));

        for (unsigned int nPrimeSeq = 1; nPrimeSeq < nPrimes; nPrimeSeq++)
        {
            const unsigned int nPrime = vPrimes[nPrimeSeq];
#ifdef USE_ROTATE
            const unsigned int nRotateBits = nPrime % nWordBits;
            for (unsigned int i = 0; i < nHalfChainLength; i++)
            {
                unsigned int nVariableMultiplier = vMultipliers[nPrimeSeq * nHalfChainLength + i];
                if (nVariableMultiplier == 0xFFFFFFFF) break;
                unsigned long lBitMask = GetBitMask(nVariableMultiplier);
                for (; nVariableMultiplier < nMaxMultiplier; nVariableMultiplier += nPrime)
                {
                    vfComposites[GetWordNum(nVariableMultiplier)] |= lBitMask;
                    lBitMask = (lBitMask << nRotateBits) | (lBitMask >> (nWordBits - nRotateBits));
                }
                vMultipliers[nPrimeSeq * nHalfChainLength + i] = nVariableMultiplier;
            }
#else
            for (unsigned int i = 0; i < nHalfChainLength; i++)
            {
                unsigned int nVariableMultiplier = vMultipliers[nPrimeSeq * nHalfChainLength + i];
                if (nVariableMultiplier == 0xFFFFFFFF) break;
                for (; nVariableMultiplier < nMaxMultiplier; nVariableMultiplier += nPrime)
                {
                    vfComposites[GetWordNum(nVariableMultiplier)] |= GetBitMask(nVariableMultiplier);
                }
                vMultipliers[nPrimeSeq * nHalfChainLength + i] = nVariableMultiplier;
            }
#endif
        }
    }

public:
    CSieveOfEratosthenes(unsigned int nSieveSize, unsigned int nBits, mpz_class& mpzHash, mpz_class& mpzFixedMultiplier, CBlockIndex* pindexPrev)
    {
        this->nSieveSize = nSieveSize;
        this->nBits = nBits;
        this->mpzHash = mpzHash;
        this->mpzFixedMultiplier = mpzFixedMultiplier;
        this->pindexPrev = pindexPrev;
        nPrimeSeq = 0;
        nCandidateCount = 0;
        nCandidateMultiplier = 0;
        nCandidatesWords = (nSieveSize + nWordBits - 1) / nWordBits;
        nCandidatesBytes = nCandidatesWords * sizeof(unsigned long);
        vfCandidates = (unsigned long *)malloc(nCandidatesBytes);
        vfCandidateBiTwin = (unsigned long *)malloc(nCandidatesBytes);
        vfCandidateCunningham1 = (unsigned long *)malloc(nCandidatesBytes);
        memset(vfCandidates, 0, nCandidatesBytes);
        memset(vfCandidateBiTwin, 0, nCandidatesBytes);
        memset(vfCandidateCunningham1, 0, nCandidatesBytes);
    }

    ~CSieveOfEratosthenes()
    {
        free(vfCandidates);
        free(vfCandidateBiTwin);
        free(vfCandidateCunningham1);
    }

    // Get total number of candidates for power test
    unsigned int GetCandidateCount()
    {
        if (nCandidateCount)
            return nCandidateCount;

        unsigned int nCandidates = 0;
#ifdef __GNUC__
        for (unsigned int i = 0; i < nCandidatesWords; i++)
        {
            nCandidates += __builtin_popcountl(vfCandidates[i]);
        }
#else
        for (unsigned int i = 0; i < nCandidatesWords; i++)
        {
            unsigned long lBits = vfCandidates[i];
            for (unsigned int j = 0; j < nWordBits; j++)
            {
                nCandidates += (lBits & 1UL);
                lBits >>= 1;
            }
        }
#endif
        nCandidateCount = nCandidates;
        return nCandidates;
    }

    // Scan for the next candidate multiplier (variable part)
    // Return values:
    //   True - found next candidate; nVariableMultiplier has the candidate
    //   False - scan complete, no more candidate and reset scan
    bool GetNextCandidateMultiplier(unsigned int& nVariableMultiplier, unsigned int& nCandidateType)
    {
        unsigned long lBits = vfCandidates[GetWordNum(nCandidateMultiplier)];
        loop
        {
            nCandidateMultiplier++;
            if (nCandidateMultiplier >= nSieveSize)
            {
                nCandidateMultiplier = 0;
                return false;
            }
            if (nCandidateMultiplier % nWordBits == 0)
            {
                lBits = vfCandidates[GetWordNum(nCandidateMultiplier)];
                if (lBits == 0)
                {
                    // Skip an entire word
                    nCandidateMultiplier += nWordBits - 1;
                    continue;
                }
            }
            if (lBits & GetBitMask(nCandidateMultiplier))
            {
                nVariableMultiplier = nCandidateMultiplier;
                if (vfCandidateBiTwin[GetWordNum(nCandidateMultiplier)] & GetBitMask(nCandidateMultiplier))
                    nCandidateType = PRIME_CHAIN_BI_TWIN;
                else if (vfCandidateCunningham1[GetWordNum(nCandidateMultiplier)] & GetBitMask(nCandidateMultiplier))
                    nCandidateType = PRIME_CHAIN_CUNNINGHAM1;
                else
                    nCandidateType = PRIME_CHAIN_CUNNINGHAM2;
                return true;
            }
        }
    }

    // Get progress percentage of the sieve
    unsigned int GetProgressPercentage();

    // Weave the sieve for the next prime in table
    // Return values:
    //   True  - weaved another prime; nComposite - number of composites removed
    //   False - sieve already completed
    bool Weave();
};

static const unsigned int nPrimorialHashFactor = 7;

inline void mpz_set_uint256(mpz_t r, uint256& u)
{
    mpz_import(r, 32 / sizeof(unsigned long), -1, sizeof(unsigned long), -1, 0, &u);
}

#endif
