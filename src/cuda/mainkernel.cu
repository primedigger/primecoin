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

//ignore intger conversion warnings here
//these are reciprocals for a quick integer division, see http://stackoverflow.com/questions/980702/efficient-cycles-wise-algorithm-to-compute-modulo-25 and http://www.hackersdelight.org/magic.htm

__device__ const unsigned int magicNumbers[] = {2147483648,2863311531,3435973837,613566757,3123612579,1321528399,4042322161,2938661835,2987803337,2369637129,138547333,3134165325,3352169597,799063683,2924233053,891408307,582368447,1126548799,128207979,3871519817,3235934265,3479467177,827945503,3088515809,1372618415,1148159575,1042467791,842937507,748664025,570128403,33818641,4196609267,125400505,1977538899,3689636335,910191745,875407347,3372735055,1645975491,2060591247,1847555765,3037324939,2878302691,356059465,1286310003,1381296015,2605477791,635578121,548696263,1200340205,423966729,2300233531,285143057,2190262207,4278255361,2090326289,4087403821,4057238479,3969356057,122276649,3885200099,3752599413,3581471101,883851791,2730666109,867122735,2348607495,815661445,1584310703,3150463117,1934560341,1830445673,1497972245,1600537411,1507204883,717696885,2826508041,692387675,2741924259,2688292489,953298231,928365853,807174829,158705489,313072787,668967819,2448800953,601483385,2385057761,37105549,2354414621,295895451,1128862041,183695139,550857529,2185907809,2160140723,4220774003,131394793,1016184499,1005038051,123374285,3905902763,966178935,1925589541,3811132159,3746206569,1854151143,3671157355,3022913755,1811386537,3587313631,3564057141,3552541609,871245347,6700417,3419942855,849699867,1683785035,3336909341,2358658289,1633746847,1624093985,12576771,795594521,1978993491,1550792141,3058446809,1754615251,3000031727,1487837115,2959654449,1561286381,726229609,1444824741,2859588109,1422395379,2794184569,1223284287,1141431359,42367125,669617313,1048953131,2659036585,2652621539,947042849,644496851,2565954791,824993719,801262729,626859537,2496053639,2490400063,1239584699,151531371,2413856483,490719877,439205483,398785651,1168450189,2322094251,18027145,2274067483,234431789,206615623,179139023,143021111,116313057,63849861,2170802819,21074423,12619885,2132903255,2128773723,4232961031,2096304343,4184630363,2072594963,2068695443,1028542215,4046040949,2015603351,2011915147,4009158169,498418689,991444209,3937373779,1958168527,3895523925,3821065605,953609391,3781639305,469475503,3724002127,1852589095,3686543597,114437097,226609981,3613842655,449514157,1789278483,3572742901,1777706755,3521254213,3493285553,2593124343,2582353285,856984901,2528999363,1703348765,3390937943,3380512307,3375323493,2435019715,3334379463,3329331197,3314277703,403935205,2139628917,3203238537,3184682485,3143707299,3121395679,1886405173,3082022783,3077709245,1843269287,764080353,3039423989,3031045149,3026873029,3014425299,1494917237,1644325765,2965641613,1620360897,1612415527,1596588647,2933986999,2910685977,1480536987,1450357997,2850321783,1383595017,1415984067,1410534481,350833321,1304073457,2785336613,2778298491,1212918129,686765539,1178643795,1366701837,340828155,1358260195,1356584365,2703163191,1343325141,2654222397,994324961,2638300247,2635138713,900598577,888352105,882250493,2573461973,816068743,2552551661,2537822569,2526161121,629371281,1254434259,705654093,77343249,638343429,1230567015,621797949,2442002505,562069161,530097445,2401991541,1190591909,431573823,2355675689,146914969,401291659,391283649,2340631459,361513923,332120039,1153132279,2299031109,284700059,255520559,218164065,213537585,1114558163,2222358015,131853551,1103373435,1101163373,105279339,96492027,79022273,66010901,44496455,2167593155,18957679,2142253537,4263738741,4251374105,4226858733,4222800299,4214706767,526333953,4190611255,1041697421,130088929,2065780419,2063841629,1029023517,1027101007,4104569773,510688169,4070380853,1009189195,1996389701,3985542829,3974737019,3625872887,3932093439,3928581073,3921575133,3907637949,1940029339,1938319309,483727069,3856244201,3846127251,1918031623,3363798941,952370401,3317402261,1885146383,117519413,1878704191,3747802737,3187885127,932975501,3709866311,3700501903,28861603,3691184651,460239275,3056552155,114580203,3648317305,3639260663,3630248875,2923820577,3603479321,3594643655,3577101677,2836036371,1778425601,3551107397,3514220145,1744564265,3475342957,2633818071,3458943383,862700375,1724048025,3440005093,2526353389,1697432077,1696122835,842860581,2427304789,3356006495,3340711365,3323042321,3310535575,2321123733,1651538307,2291399625,2276637503,3278454351,3273573883,1635569547,3266280365,3259019275,812346973,2194219737,2189435965,3235047085,3223192753,201302019,2123196165,399967853,2095220153,198682983,3167480383,2012736557,3151591911,393103907,785085061,1981231793,1945609521,1914770101,775122755,3093947599,385658235,1862605699,1853996719,1815552345,1798619835,3036276501,1765034787,3023751469,1735950101,1502578241,748221591,2978697265,1654368533,2968644287,1481322503,1626333965,2933008677,1465526995,2921319503,2913578345,1524478965,2896309853,723125043,1442455399,1452234941,1433989733,2856801891,2853095369,2847553585,2829235453,2820164483,2818357265,2803982475,2780933615,1388710613,2775668357,1235427563,1225015775,1218096335,2746204503,2741069811,2734253349,1166748335,1362045993,2705657651,1113021037,1350336663,2699015963,2689114345,2666290701,2664675257,332480081,1015079805,1005480443,999100127,989558401,986385465,967427573,1314026445,942361089,327138241,923717381,2607795145,2595483335,162122033,2581770773,2577232061,2562217601,2550331407,1272214785,2541488883,785074877,2537090575,776288411,744329767,2513887689,2505295649,156313851,692893789,1246258575,684423037,310684269,673173355,1239934173,650825801,648046485,631435965};
__device__ const unsigned int magicShifts[] = {0,1,2,3,3,2,4,5,4,4,5,6,5,3,5,6,3,4,1,6,7,6,4,6,7,7,7,7,7,7,7,7,2,6,7,5,5,7,6,8,8,7,7,4,8,6,7,8,8,6,8,7,4,7,8,7,8,8,8,3,8,8,8,6,9,6,9,6,7,8,9,9,7,9,9,6,8,6,8,8,9,9,9,4,5,9,8,6,8,2,8,9,7,9,6,8,8,9,4,7,7,4,9,7,8,9,9,8,9,10,8,9,9,9,7,0,9,7,8,9,10,8,8,1,7,10,8,9,10,9,8,9,10,7,8,9,8,9,10,10,3,7,10,9,9,10,7,9,10,10,7,9,9,8,5,9,10,10,10,8,9,2,9,10,10,10,10,10,10,9,10,10,9,9,10,9,10,9,9,8,10,9,9,10,7,8,10,9,10,10,8,10,7,10,9,10,5,6,10,7,9,10,9,10,10,11,11,8,11,9,10,10,10,11,10,10,10,7,11,10,10,10,10,11,10,10,11,8,10,10,10,10,9,11,10,11,11,11,10,10,11,11,10,11,9,9,7,11,10,10,11,8,11,9,7,9,9,10,9,10,11,10,10,11,11,11,10,11,10,10,10,8,9,11,5,11,9,11,10,11,11,10,9,11,10,6,11,11,10,11,11,9,10,7,11,11,11,9,10,11,9,9,11,11,11,11,11,10,11,10,11,11,11,11,11,8,11,9,6,10,10,9,9,11,8,11,9,10,11,11,12,11,11,11,11,10,10,8,11,11,10,12,9,12,10,6,10,11,12,9,11,11,4,11,8,12,6,11,11,11,12,11,11,11,12,10,11,11,10,11,12,11,9,10,11,12,10,10,9,12,11,11,11,11,12,10,12,12,11,11,10,11,11,9,12,12,11,11,7,12,8,12,7,11,12,11,8,9,12,12,12,9,11,8,12,12,12,12,11,12,11,12,10,9,11,12,11,10,12,11,10,11,11,12,11,9,10,12,10,11,11,11,11,11,11,11,11,10,11,12,12,12,11,11,11,12,10,11,12,10,11,11,11,11,8,12,12,12,12,12,12,10,12,8,12,11,11,7,11,11,11,11,10,11,12,11,12,12,11,11,7,12,10,12,8,12,10,12,12,12};
__device__ const unsigned int magicAdds[] = {0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,1,0,1,1,1};

__device__ const unsigned int primes_len = 1229;
__device__ const unsigned int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973};

__device__ unsigned int fastMod(unsigned int N, unsigned int p, unsigned int p_i, unsigned int index)
{
	unsigned int magic = magicNumbers[p_i];
	unsigned int shift = magicShifts[p_i];
	unsigned int div = __umulhi(N,magic);
	unsigned int add = magicAdds[p_i]*N;

	//this can overflow?: div += add; we could also do add with carry in asm:
	/*

	unsigned int carry = 0;
	unsigned int temp = 0;

	asm volatile ("add.cc.u32 %0, %2, %3;"
      	    	      "addc.u32   %1, 0, 0;"
      			: "=r"(temp), "=r"(carry)
      			: "r"(div), "r"(add));

	carry <<= 32-shift;
	div = temp;*/

	div += add;
	div >>= shift;
	//div |= carry;

	#ifdef MOD_DEBUG
	if(index==0 && N/p != div)
		printf("fast mod not working, N:%i, Divisor %i, magic %i, shift %i, should be %i is %i\n",N,p,magic,shift,N/p,div);
	#endif
	//else if (index==0 && N/p == div)
	//	printf("fast mod working, N:%i, Divisor %i, magic %i, shift %i, carry %i, should be %i is %i\n",N,p,magic,shift,carry,N/p,div);
		
	return N - div*p;
}

__device__ unsigned int fastMpzPrimeMod(mpz_cuda_t *N, unsigned int p_i, unsigned int index)
{
	unsigned int p = primes[p_i];
	//bool found_composite = false;
	unsigned int mod = 0;
	for (int i=N->capacity-1; i >= 0; i--)
	{
		unsigned int digit = N->digits[i];
		unsigned int hi = digit >> 16;
		unsigned int low = digit & 0x0000ffff;
		
		//mod = fastMod((mod * 0x10000 +hi),p,p_i,index);
		//mod = fastMod((mod * 0x10000 +low),p,p_i,index);

		mod = (mod * 0x10000 +hi) % p;
		mod = (mod * 0x10000 +low) % p;
	}
	
	/*if(index==0)
	{
		printf("N: ");
		mpz_print(N);
		printf(" mod %i = %i ",p,mod);
	}*/

	return mod;
}

#define MAXCHAIN 10

__device__ int fastModPrimeChecks(mpz_cuda_t *N,unsigned int index)
{
	/*if(index == 0)
	{
		printf("testing N:");
		mpz_print(N);
	}*/
	bool isLeft = (index % 2 ==0);
	int factors = 11;
	for (int i=0; i < 30; i++)
	{
		int p = primes[i];
		int mod = fastMpzPrimeMod(N, i, index);
		if(mod==0) factors = 0;
		for(int j=1; j <= MAXCHAIN; j++)
		{
			mod = mod*2;
			if(isLeft)
			{
				mod -= 1;
			}
			else
			{
				mod += 1;
			}
			mod = mod % p;
			if(mod==0)
			{
				if(j < factors)
				{
					if(index==1)
					{
						printf("Found better divisor:%i in chain %i previous %i\n",p,j,factors);
						printf("For N:");
						mpz_print(N);
						printf("\n");
					}
					factors = j;
				}
	
			}
		}
	}
	if(index==1) { printf("factors is %i \n",factors); }
	return factors;
}

__global__ void runPrimeCandidateSearch(cudaCandidate *candidates, char *result, unsigned int num_candidates)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int threadIndex = threadIdx.x;

	__shared__ mpz_cuda_t mpzN[192];
	//__shared__ mpz_cuda_t mpzN_copy1[192];
//	__shared__ mpz_cuda_t mpzN_copy2[192];

	//check bounds
	if (index < 2*num_candidates)
	{
		cudaCandidate candidate = candidates[index/2];	
	
		//mpz_cuda_t mpzN;
		mpz_init(&mpzN[threadIndex]);
		mpz_set(&mpzN[threadIndex],&candidate.chainOrigin);
		
		//sloppy add
		/*if(index % 2 == 0)
			mpzN[threadIndex].digits[0] -= 1;
		else
			mpzN[threadIndex].digits[0] += 1;*/	
		mpz_addeq_i(&mpzN[threadIndex], index % 2 == 0 ? -1 : 1);

		//int testresult = 0;

		
		/*if(fastModPrimeChecks(&mpzN[threadIndex],index) == true)
		{
			testresult = 1;
			if (index==0) printf("\nDetected composite in base prime!\n");
		}*/

		int myresult = fastModPrimeChecks(&mpzN[threadIndex],index);
		
		//mpz_cuda_t mpzN_copy;
		//mpz_init(&mpzN_copy1[threadIndex]);
		//mpz_init(&mpzN_copy2[threadIndex]);

		//check partial chain
		/*for(int i=0; i < 1; i++)
		{
			if (index==0)
			{
				printf("N:");
				mpz_print( &mpzN[threadIndex] );
			}

			mpz_set(&mpzN_copy1[threadIndex],&mpzN[threadIndex]);
			//mpz_mult_u(&mpzN[threadIndex],&mpzN_copy1[threadIndex],2);
			mpz_add(&mpzN[threadIndex],&mpzN_copy1[threadIndex],&mpzN_copy1[threadIndex]);

			if (index==0)
			{
				printf("N+N:");
				mpz_print( &mpzN[threadIndex] );
			}				
			
			//mpz_set(&mpzN_copy2[threadIndex],&mpzN[threadIndex]);
			//mpz_add(&mpzN[threadIndex],&mpzN_copy1[threadIndex],&mpzN_copy2[threadIndex]);
			
			//sloppy add
			mpzN[threadIndex].digits[0] += ((index % 2 == 0) ? -1 : 1);
			if(fastModPrimeChecks(&mpzN[threadIndex],index) == true)
			{
				testresult = 1;
				if (index==0) 
				{
					printf("\nDetected composite in chain!\n");
				}
			}
		}*/

		//printf("%i ",testresult);

		//__syncthreads;

		/*if(testresult==0)
		{
			printf("[GPU] Possible chain in: %i!\n",index);
		}*/

		result[index] = myresult;
		//printf("done");
	}
}

__global__ void runPrimeCandidateSearchOld(cudaCandidate *candidates, char *result, unsigned int num_candidates)
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
