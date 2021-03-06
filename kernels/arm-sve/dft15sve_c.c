/*

     FFTE: A FAST FOURIER TRANSFORM PACKAGE

     (C) COPYRIGHT SOFTWARE, 2000-2004, 2008-2014, ALL RIGHTS RESERVED
                BY
         DAISUKE TAKAHASHI
         FACULTY OF ENGINEERING, INFORMATION AND SYSTEMS
         UNIVERSITY OF TSUKUBA
         1-1-1 TENNODAI, TSUKUBA, IBARAKI 305-8573, JAPAN
         E-MAIL: daisuke@cs.tsukuba.ac.jp


     WRITTEN BY DAISUKE TAKAHASHI

     THIS KERNEL WAS GENERATED BY SPIRAL 8.2.0a03
*/

#include <arm_sve.h>

void dft15c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    int l1, m1;
    svfloat64x2_t s1001, s1004, s1007, s1010, s1013, s1016, s1019, s977, 
            s980, s983, s986, s989, s992, s995, s998, svex2_16, 
            svex2_17, svex2_18, svex2_19, svex2_20, svex2_21, svex2_22, svex2_23, svex2_24, 
            svex2_25, svex2_26, svex2_27, svex2_28, svex2_29, svex2_30;
    svfloat64_t s1000, s1002, s1003, s1005, s1006, s1008, s1009, s1011, 
            s1012, s1014, s1015, s1017, s1018, s1020, s1021, s1022, 
            s1023, s1024, s1025, s1026, s1027, s1028, s1029, s1030, 
            s1031, s1032, s1033, s1034, s1035, s1036, s1037, s1038, 
            s1039, s1040, s1041, s1042, s1043, s1044, s1045, s1046, 
            s1047, s1048, s1049, s1050, s1051, s1052, s1053, s1054, 
            s1055, s1056, s1057, s1058, s1059, s1060, s1061, s1062, 
            s1063, s1064, s1065, s1066, s1067, s1068, s1069, s1070, 
            s1071, s1072, s1073, s1074, s1075, s1076, s1077, s1078, 
            s1079, s1080, s1081, s1082, s1083, s1084, s1085, s1086, 
            s1087, s1088, s1089, s1090, s1091, s1092, s1093, s978, 
            s979, s981, s982, s984, s985, s987, s988, s990, 
            s991, s993, s994, s996, s997, s999, t2300, t2301, 
            t2302, t2303, t2304, t2305, t2306, t2307, t2308, t2309, 
            t2310, t2311, t2312, t2313, t2314, t2315, t2316, t2317, 
            t2318, t2319, t2320, t2321, t2322, t2323, t2324, t2325, 
            t2326, t2327, t2328, t2329, t2330, t2331, t2332, t2333, 
            t2334, t2335, t2336, t2337, t2338, t2339, t2340, t2341, 
            t2342, t2343, t2344, t2345, t2346, t2347, t2348, t2349, 
            t2350, t2351, t2352, t2353, t2354, t2355, t2356, t2357, 
            t2358, t2359, t2360, t2361, t2362, t2363, t2364, t2365, 
            t2366, t2367, t2368, t2369, t2370, t2371, t2372, t2373, 
            t2374, t2375, t2376, t2377, t2378, t2379, t2380, t2381, 
            t2382, t2383, t2384, t2385, t2386, t2387, t2388, t2389, 
            t2390, t2391, t2392, t2393, t2394, t2395, t2396, t2397, 
            t2398, t2399, t2400, t2401, t2402, t2403, t2404, t2405, 
            t2406, t2407, t2408, t2409, t2410, t2411, t2412, t2413, 
            t2414, t2415, t2416, t2417, t2418, t2419, t2420, t2421;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s977 = svld2_f64(pg1, (X + ((2)*(k1))));
        s978 = s977.v0;
        s979 = s977.v1;
        s980 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s981 = s980.v0;
        s982 = s980.v1;
        s983 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s984 = s983.v0;
        s985 = s983.v1;
        s986 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s987 = s986.v0;
        s988 = s986.v1;
        s989 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s990 = s989.v0;
        s991 = s989.v1;
        s992 = svld2_f64(pg1, (X + ((2)*((k1 + ((((5)*(l1)))*(m1)))))));
        s993 = s992.v0;
        s994 = s992.v1;
        s995 = svld2_f64(pg1, (X + ((2)*((k1 + ((((6)*(l1)))*(m1)))))));
        s996 = s995.v0;
        s997 = s995.v1;
        s998 = svld2_f64(pg1, (X + ((2)*((k1 + ((((7)*(l1)))*(m1)))))));
        s999 = s998.v0;
        s1000 = s998.v1;
        s1001 = svld2_f64(pg1, (X + ((2)*((k1 + ((((8)*(l1)))*(m1)))))));
        s1002 = s1001.v0;
        s1003 = s1001.v1;
        s1004 = svld2_f64(pg1, (X + ((2)*((k1 + ((((9)*(l1)))*(m1)))))));
        s1005 = s1004.v0;
        s1006 = s1004.v1;
        s1007 = svld2_f64(pg1, (X + ((2)*((k1 + ((((10)*(l1)))*(m1)))))));
        s1008 = s1007.v0;
        s1009 = s1007.v1;
        s1010 = svld2_f64(pg1, (X + ((2)*((k1 + ((((11)*(l1)))*(m1)))))));
        s1011 = s1010.v0;
        s1012 = s1010.v1;
        s1013 = svld2_f64(pg1, (X + ((2)*((k1 + ((((12)*(l1)))*(m1)))))));
        s1014 = s1013.v0;
        s1015 = s1013.v1;
        s1016 = svld2_f64(pg1, (X + ((2)*((k1 + ((((13)*(l1)))*(m1)))))));
        s1017 = s1016.v0;
        s1018 = s1016.v1;
        s1019 = svld2_f64(pg1, (X + ((2)*((k1 + ((((14)*(l1)))*(m1)))))));
        s1020 = s1019.v0;
        s1021 = s1019.v1;
        t2300 = svadd_f64_x(pg1, s987, s1014);
        t2301 = svadd_f64_x(pg1, s988, s1015);
        t2302 = svsub_f64_x(pg1, s987, s1014);
        t2303 = svsub_f64_x(pg1, s988, s1015);
        t2304 = svadd_f64_x(pg1, s996, s1005);
        t2305 = svadd_f64_x(pg1, s997, s1006);
        t2306 = svsub_f64_x(pg1, s996, s1005);
        t2307 = svsub_f64_x(pg1, s997, s1006);
        t2308 = svadd_f64_x(pg1, t2300, t2304);
        t2309 = svadd_f64_x(pg1, t2301, t2305);
        t2310 = svadd_f64_x(pg1, t2302, t2307);
        t2311 = svsub_f64_x(pg1, t2303, t2306);
        t2312 = svsub_f64_x(pg1, t2302, t2307);
        t2313 = svadd_f64_x(pg1, t2303, t2306);
        t2314 = svadd_f64_x(pg1, s978, t2308);
        t2315 = svadd_f64_x(pg1, s979, t2309);
        t2316 = svmls_n_f64_x(pg1, s978, t2308, 0.25);
        t2317 = svmls_n_f64_x(pg1, s979, t2309, 0.25);
        s1066 = svmla_n_f64_x(pg1, t2310, t2311, 1.6180339887498947);
        s1022 = svmul_n_f64_x(pg1, s1066, 0.29389262614623657);
        s1067 = svmls_n_f64_x(pg1, t2311, t2310, 1.6180339887498947);
        s1023 = svmul_n_f64_x(pg1, s1067, 0.29389262614623657);
        s1024 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2300, t2304), 0.55901699437494745);
        s1025 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2301, t2305), 0.55901699437494745);
        s1068 = svmls_n_f64_x(pg1, t2313, t2312, 0.61803398874989479);
        s1026 = svmul_n_f64_x(pg1, s1068, 0.47552825814757682);
        s1069 = svmla_n_f64_x(pg1, t2312, t2313, 0.61803398874989479);
        s1027 = svmul_n_f64_x(pg1, s1069, 0.47552825814757682);
        t2318 = svadd_f64_x(pg1, t2316, s1024);
        t2319 = svadd_f64_x(pg1, t2317, s1025);
        t2320 = svsub_f64_x(pg1, t2316, s1024);
        t2321 = svsub_f64_x(pg1, t2317, s1025);
        t2322 = svadd_f64_x(pg1, s1022, s1026);
        t2323 = svsub_f64_x(pg1, s1023, s1027);
        t2324 = svsub_f64_x(pg1, s1022, s1026);
        t2325 = svadd_f64_x(pg1, s1023, s1027);
        t2326 = svadd_f64_x(pg1, t2318, t2322);
        t2327 = svadd_f64_x(pg1, t2319, t2323);
        t2328 = svsub_f64_x(pg1, t2318, t2322);
        t2329 = svsub_f64_x(pg1, t2319, t2323);
        t2330 = svadd_f64_x(pg1, t2320, t2325);
        t2331 = svsub_f64_x(pg1, t2321, t2324);
        t2332 = svsub_f64_x(pg1, t2320, t2325);
        t2333 = svadd_f64_x(pg1, t2321, t2324);
        t2334 = svadd_f64_x(pg1, s990, s1017);
        t2335 = svadd_f64_x(pg1, s991, s1018);
        t2336 = svsub_f64_x(pg1, s990, s1017);
        t2337 = svsub_f64_x(pg1, s991, s1018);
        t2338 = svadd_f64_x(pg1, s999, s1008);
        t2339 = svadd_f64_x(pg1, s1000, s1009);
        t2340 = svsub_f64_x(pg1, s999, s1008);
        t2341 = svsub_f64_x(pg1, s1000, s1009);
        t2342 = svadd_f64_x(pg1, t2334, t2338);
        t2343 = svadd_f64_x(pg1, t2335, t2339);
        t2344 = svadd_f64_x(pg1, t2336, t2341);
        t2345 = svsub_f64_x(pg1, t2337, t2340);
        t2346 = svsub_f64_x(pg1, t2336, t2341);
        t2347 = svadd_f64_x(pg1, t2337, t2340);
        t2348 = svadd_f64_x(pg1, s981, t2342);
        t2349 = svadd_f64_x(pg1, s982, t2343);
        t2350 = svmls_n_f64_x(pg1, s981, t2342, 0.25);
        t2351 = svmls_n_f64_x(pg1, s982, t2343, 0.25);
        s1070 = svmla_n_f64_x(pg1, t2344, t2345, 1.6180339887498947);
        s1028 = svmul_n_f64_x(pg1, s1070, 0.29389262614623657);
        s1071 = svmls_n_f64_x(pg1, t2345, t2344, 1.6180339887498947);
        s1029 = svmul_n_f64_x(pg1, s1071, 0.29389262614623657);
        s1030 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2334, t2338), 0.55901699437494745);
        s1031 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2335, t2339), 0.55901699437494745);
        s1072 = svmls_n_f64_x(pg1, t2347, t2346, 0.61803398874989479);
        s1032 = svmul_n_f64_x(pg1, s1072, 0.47552825814757682);
        s1073 = svmla_n_f64_x(pg1, t2346, t2347, 0.61803398874989479);
        s1033 = svmul_n_f64_x(pg1, s1073, 0.47552825814757682);
        t2352 = svadd_f64_x(pg1, t2350, s1030);
        t2353 = svadd_f64_x(pg1, t2351, s1031);
        t2354 = svsub_f64_x(pg1, t2350, s1030);
        t2355 = svsub_f64_x(pg1, t2351, s1031);
        t2356 = svadd_f64_x(pg1, s1028, s1032);
        t2357 = svsub_f64_x(pg1, s1029, s1033);
        t2358 = svsub_f64_x(pg1, s1028, s1032);
        t2359 = svadd_f64_x(pg1, s1029, s1033);
        t2360 = svadd_f64_x(pg1, t2352, t2356);
        t2361 = svadd_f64_x(pg1, t2353, t2357);
        t2362 = svsub_f64_x(pg1, t2352, t2356);
        t2363 = svsub_f64_x(pg1, t2353, t2357);
        s1074 = svmla_n_f64_x(pg1, t2360, t2361, 0.4452286853085361);
        s1034 = svmul_n_f64_x(pg1, s1074, 0.91354545764260087);
        s1075 = svmls_n_f64_x(pg1, t2361, t2360, 0.4452286853085361);
        s1035 = svmul_n_f64_x(pg1, s1075, 0.91354545764260087);
        s1076 = svmls_n_f64_x(pg1, t2363, t2362, 0.10510423526567646);
        s1036 = svmul_n_f64_x(pg1, s1076, 0.99452189536827329);
        s1077 = svmla_n_f64_x(pg1, t2362, t2363, 0.10510423526567646);
        s1037 = svmul_n_f64_x(pg1, s1077, 0.99452189536827329);
        t2364 = svadd_f64_x(pg1, t2354, t2359);
        t2365 = svsub_f64_x(pg1, t2355, t2358);
        t2366 = svsub_f64_x(pg1, t2354, t2359);
        t2367 = svadd_f64_x(pg1, t2355, t2358);
        s1078 = svmla_n_f64_x(pg1, t2364, t2365, 1.1106125148291928);
        s1038 = svmul_n_f64_x(pg1, s1078, 0.66913060635885824);
        s1079 = svmls_n_f64_x(pg1, t2365, t2364, 1.1106125148291928);
        s1039 = svmul_n_f64_x(pg1, s1079, 0.66913060635885824);
        s1080 = svmla_n_f64_x(pg1, t2366, t2367, 3.0776835371752536);
        s1040 = svmul_n_f64_x(pg1, s1080, 0.3090169943749474);
        s1081 = svmls_n_f64_x(pg1, t2367, t2366, 3.0776835371752536);
        s1041 = svmul_n_f64_x(pg1, s1081, 0.3090169943749474);
        t2368 = svadd_f64_x(pg1, s993, s1020);
        t2369 = svadd_f64_x(pg1, s994, s1021);
        t2370 = svsub_f64_x(pg1, s993, s1020);
        t2371 = svsub_f64_x(pg1, s994, s1021);
        t2372 = svadd_f64_x(pg1, s1002, s1011);
        t2373 = svadd_f64_x(pg1, s1003, s1012);
        t2374 = svsub_f64_x(pg1, s1002, s1011);
        t2375 = svsub_f64_x(pg1, s1003, s1012);
        t2376 = svadd_f64_x(pg1, t2368, t2372);
        t2377 = svadd_f64_x(pg1, t2369, t2373);
        t2378 = svadd_f64_x(pg1, t2370, t2375);
        t2379 = svsub_f64_x(pg1, t2371, t2374);
        t2380 = svsub_f64_x(pg1, t2370, t2375);
        t2381 = svadd_f64_x(pg1, t2371, t2374);
        t2382 = svadd_f64_x(pg1, s984, t2376);
        t2383 = svadd_f64_x(pg1, s985, t2377);
        t2384 = svmls_n_f64_x(pg1, s984, t2376, 0.25);
        t2385 = svmls_n_f64_x(pg1, s985, t2377, 0.25);
        s1082 = svmla_n_f64_x(pg1, t2378, t2379, 1.6180339887498947);
        s1042 = svmul_n_f64_x(pg1, s1082, 0.29389262614623657);
        s1083 = svmls_n_f64_x(pg1, t2379, t2378, 1.6180339887498947);
        s1043 = svmul_n_f64_x(pg1, s1083, 0.29389262614623657);
        s1044 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2368, t2372), 0.55901699437494745);
        s1045 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2369, t2373), 0.55901699437494745);
        s1084 = svmls_n_f64_x(pg1, t2381, t2380, 0.61803398874989479);
        s1046 = svmul_n_f64_x(pg1, s1084, 0.47552825814757682);
        s1085 = svmla_n_f64_x(pg1, t2380, t2381, 0.61803398874989479);
        s1047 = svmul_n_f64_x(pg1, s1085, 0.47552825814757682);
        t2386 = svadd_f64_x(pg1, t2384, s1044);
        t2387 = svadd_f64_x(pg1, t2385, s1045);
        t2388 = svsub_f64_x(pg1, t2384, s1044);
        t2389 = svsub_f64_x(pg1, t2385, s1045);
        t2390 = svadd_f64_x(pg1, s1042, s1046);
        t2391 = svsub_f64_x(pg1, s1043, s1047);
        t2392 = svsub_f64_x(pg1, s1042, s1046);
        t2393 = svadd_f64_x(pg1, s1043, s1047);
        t2394 = svadd_f64_x(pg1, t2386, t2390);
        t2395 = svadd_f64_x(pg1, t2387, t2391);
        t2396 = svsub_f64_x(pg1, t2386, t2390);
        t2397 = svsub_f64_x(pg1, t2387, t2391);
        s1086 = svmla_n_f64_x(pg1, t2394, t2395, 1.1106125148291928);
        s1048 = svmul_n_f64_x(pg1, s1086, 0.66913060635885824);
        s1087 = svmls_n_f64_x(pg1, t2395, t2394, 1.1106125148291928);
        s1049 = svmul_n_f64_x(pg1, s1087, 0.66913060635885824);
        s1088 = svmla_n_f64_x(pg1, t2396, t2397, 0.2125565616700221);
        s1050 = svmul_n_f64_x(pg1, s1088, 0.97814760073380569);
        s1089 = svmls_n_f64_x(pg1, t2396, t2397, 4.7046301094784546);
        s1051 = svmul_n_f64_x(pg1, s1089, 0.20791169081775931);
        t2398 = svadd_f64_x(pg1, t2388, t2393);
        t2399 = svsub_f64_x(pg1, t2389, t2392);
        t2400 = svsub_f64_x(pg1, t2388, t2393);
        t2401 = svadd_f64_x(pg1, t2389, t2392);
        s1090 = svmls_n_f64_x(pg1, t2399, t2398, 0.10510423526567646);
        s1052 = svmul_n_f64_x(pg1, s1090, 0.99452189536827329);
        s1091 = svmla_n_f64_x(pg1, t2398, t2399, 0.10510423526567646);
        s1053 = svmul_n_f64_x(pg1, s1091, 0.99452189536827329);
        s1092 = svmls_n_f64_x(pg1, t2401, t2400, 1.3763819204711736);
        s1054 = svmul_n_f64_x(pg1, s1092, 0.58778525229247314);
        s1093 = svmla_n_f64_x(pg1, t2400, t2401, 1.3763819204711736);
        s1055 = svmul_n_f64_x(pg1, s1093, 0.58778525229247314);
        t2402 = svadd_f64_x(pg1, t2348, t2382);
        t2403 = svadd_f64_x(pg1, t2349, t2383);
        t2404 = svmls_n_f64_x(pg1, t2314, t2402, 0.5);
        t2405 = svmls_n_f64_x(pg1, t2315, t2403, 0.5);
        s1056 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2349, t2383), 0.8660254037844386);
        s1057 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2348, t2382), 0.8660254037844386);
        t2406 = svadd_f64_x(pg1, s1034, s1048);
        t2407 = svadd_f64_x(pg1, s1035, s1049);
        t2408 = svmls_n_f64_x(pg1, t2326, t2406, 0.5);
        t2409 = svmls_n_f64_x(pg1, t2327, t2407, 0.5);
        s1058 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1035, s1049), 0.8660254037844386);
        s1059 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1034, s1048), 0.8660254037844386);
        t2410 = svadd_f64_x(pg1, s1038, s1052);
        t2411 = svsub_f64_x(pg1, s1039, s1053);
        t2412 = svmls_n_f64_x(pg1, t2330, t2410, 0.5);
        t2413 = svmls_n_f64_x(pg1, t2331, t2411, 0.5);
        s1060 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1039, s1053), 0.8660254037844386);
        s1061 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1038, s1052), 0.8660254037844386);
        t2414 = svadd_f64_x(pg1, s1040, s1054);
        t2415 = svsub_f64_x(pg1, s1041, s1055);
        t2416 = svmls_n_f64_x(pg1, t2332, t2414, 0.5);
        t2417 = svmls_n_f64_x(pg1, t2333, t2415, 0.5);
        s1062 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1041, s1055), 0.8660254037844386);
        s1063 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1040, s1054), 0.8660254037844386);
        t2418 = svsub_f64_x(pg1, s1036, s1050);
        t2419 = svsub_f64_x(pg1, s1051, s1037);
        t2420 = svmls_n_f64_x(pg1, t2328, t2418, 0.5);
        t2421 = svmls_n_f64_x(pg1, t2329, t2419, 0.5);
        s1064 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1037, s1051), 0.8660254037844386);
        s1065 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1036, s1050), 0.8660254037844386);
        svex2_16.v0 = svadd_f64_x(pg1, t2314, t2402);
        svex2_16.v1 = svadd_f64_x(pg1, t2315, t2403);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_16);
        svex2_17.v0 = svadd_f64_x(pg1, t2326, t2406);
        svex2_17.v1 = svadd_f64_x(pg1, t2327, t2407);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_17);
        svex2_18.v0 = svadd_f64_x(pg1, t2330, t2410);
        svex2_18.v1 = svadd_f64_x(pg1, t2331, t2411);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_18);
        svex2_19.v0 = svadd_f64_x(pg1, t2332, t2414);
        svex2_19.v1 = svadd_f64_x(pg1, t2333, t2415);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_19);
        svex2_20.v0 = svadd_f64_x(pg1, t2328, t2418);
        svex2_20.v1 = svadd_f64_x(pg1, t2329, t2419);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_20);
        svex2_21.v0 = svadd_f64_x(pg1, t2404, s1056);
        svex2_21.v1 = svsub_f64_x(pg1, t2405, s1057);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((5)*(m1)))))), svex2_21);
        svex2_22.v0 = svadd_f64_x(pg1, t2408, s1058);
        svex2_22.v1 = svsub_f64_x(pg1, t2409, s1059);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((6)*(m1)))))), svex2_22);
        svex2_23.v0 = svadd_f64_x(pg1, t2412, s1060);
        svex2_23.v1 = svsub_f64_x(pg1, t2413, s1061);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((7)*(m1)))))), svex2_23);
        svex2_24.v0 = svadd_f64_x(pg1, t2416, s1062);
        svex2_24.v1 = svsub_f64_x(pg1, t2417, s1063);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((8)*(m1)))))), svex2_24);
        svex2_25.v0 = svsub_f64_x(pg1, t2420, s1064);
        svex2_25.v1 = svsub_f64_x(pg1, t2421, s1065);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((9)*(m1)))))), svex2_25);
        svex2_26.v0 = svsub_f64_x(pg1, t2404, s1056);
        svex2_26.v1 = svadd_f64_x(pg1, t2405, s1057);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((10)*(m1)))))), svex2_26);
        svex2_27.v0 = svsub_f64_x(pg1, t2408, s1058);
        svex2_27.v1 = svadd_f64_x(pg1, t2409, s1059);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((11)*(m1)))))), svex2_27);
        svex2_28.v0 = svsub_f64_x(pg1, t2412, s1060);
        svex2_28.v1 = svadd_f64_x(pg1, t2413, s1061);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((12)*(m1)))))), svex2_28);
        svex2_29.v0 = svsub_f64_x(pg1, t2416, s1062);
        svex2_29.v1 = svadd_f64_x(pg1, t2417, s1063);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((13)*(m1)))))), svex2_29);
        svex2_30.v0 = svadd_f64_x(pg1, t2420, s1064);
        svex2_30.v1 = svadd_f64_x(pg1, t2421, s1065);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((14)*(m1)))))), svex2_30);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}
