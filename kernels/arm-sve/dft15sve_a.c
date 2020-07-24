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

void dft15a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int l1;
    float64_t  *a1724;
    svfloat64x2_t s1091, s1094, s1097, s1100, s1103, s1106, s1109, s1112, 
            s1115, s1118, s1121, s1124, s1127, s1130, s1133;
    svfloat64_t a1725, a1726, a1727, a1728, a1729, a1730, a1731, a1732, 
            a1733, a1734, a1735, a1736, a1737, a1738, a1739, a1740, 
            a1741, a1742, a1743, a1744, a1745, a1746, a1747, a1748, 
            a1749, a1750, a1751, a1752, s1092, s1093, s1095, s1096, 
            s1098, s1099, s1101, s1102, s1104, s1105, s1107, s1108, 
            s1110, s1111, s1113, s1114, s1116, s1117, s1119, s1120, 
            s1122, s1123, s1125, s1126, s1128, s1129, s1131, s1132, 
            s1134, s1135, s1136, s1137, s1138, s1139, s1140, s1141, 
            s1142, s1143, s1144, s1145, s1146, s1147, s1148, s1149, 
            s1150, s1151, s1152, s1153, s1154, s1155, s1156, s1157, 
            s1158, s1159, s1160, s1161, s1162, s1163, s1164, s1165, 
            s1166, s1167, s1168, s1169, s1170, s1171, s1172, s1173, 
            s1174, s1175, s1176, s1177, s1178, s1179, s1180, s1181, 
            s1182, s1183, s1184, s1185, s1186, s1187, s1188, s1189, 
            s1190, s1191, s1192, s1193, s1194, s1195, s1196, s1197, 
            s1198, s1199, s1200, s1201, s1202, s1203, s1204, s1205, 
            s1206, s1207, s1208, s1209, s1210, s1211, s1212, s1213, 
            s1214, s1215, s1216, s1217, s1218, s1219, s1220, s1221, 
            s1222, s1223, s1224, s1225, s1226, s1227, s1228, s1229, 
            s1230, s1231, s1232, s1233, s1234, s1235, s1236, s1237, 
            s1238, s1239, s1240, s1241, s1242, s1243, s1244, s1245, 
            s1246, s1247, s1248, s1249, s1250, s1251, s1252, s1253, 
            s1254, s1255, s1256, s1257, s1258, s1259, s1260, s1261, 
            s1262, s1263, s1264, s1265, t2301, t2302, t2303, t2304, 
            t2305, t2306, t2307, t2308, t2309, t2310, t2311, t2312, 
            t2313, t2314, t2315, t2316, t2317, t2318, t2319, t2320, 
            t2321, t2322, t2323, t2324, t2325, t2326, t2327, t2328, 
            t2329, t2330, t2331, t2332, t2333, t2334, t2335, t2336, 
            t2337, t2338, t2339, t2340, t2341, t2342, t2343, t2344, 
            t2345, t2346, t2347, t2348, t2349, t2350, t2351, t2352, 
            t2353, t2354, t2355, t2356, t2357, t2358, t2359, t2360, 
            t2361, t2362, t2363, t2364, t2365, t2366, t2367, t2368, 
            t2369, t2370, t2371, t2372, t2373, t2374, t2375, t2376, 
            t2377, t2378, t2379, t2380, t2381, t2382, t2383, t2384, 
            t2385, t2386, t2387, t2388, t2389, t2390, t2391, t2392, 
            t2393, t2394, t2395, t2396, t2397, t2398, t2399, t2400, 
            t2401, t2402, t2403, t2404, t2405, t2406, t2407, t2408, 
            t2409, t2410, t2411, t2412, t2413, t2414, t2415, t2416, 
            t2417, t2418, t2419, t2420, t2421, t2422;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s1091 = svld2_f64(pg1, (X + ((2)*(j1))));
        s1092 = s1091.v0;
        s1093 = s1091.v1;
        s1094 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s1095 = s1094.v0;
        s1096 = s1094.v1;
        s1097 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s1098 = s1097.v0;
        s1099 = s1097.v1;
        s1100 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s1101 = s1100.v0;
        s1102 = s1100.v1;
        s1103 = svld2_f64(pg1, (X + ((2)*((j1 + ((4)*(l1)))))));
        s1104 = s1103.v0;
        s1105 = s1103.v1;
        s1106 = svld2_f64(pg1, (X + ((2)*((j1 + ((5)*(l1)))))));
        s1107 = s1106.v0;
        s1108 = s1106.v1;
        s1109 = svld2_f64(pg1, (X + ((2)*((j1 + ((6)*(l1)))))));
        s1110 = s1109.v0;
        s1111 = s1109.v1;
        s1112 = svld2_f64(pg1, (X + ((2)*((j1 + ((7)*(l1)))))));
        s1113 = s1112.v0;
        s1114 = s1112.v1;
        s1115 = svld2_f64(pg1, (X + ((2)*((j1 + ((8)*(l1)))))));
        s1116 = s1115.v0;
        s1117 = s1115.v1;
        s1118 = svld2_f64(pg1, (X + ((2)*((j1 + ((9)*(l1)))))));
        s1119 = s1118.v0;
        s1120 = s1118.v1;
        s1121 = svld2_f64(pg1, (X + ((2)*((j1 + ((10)*(l1)))))));
        s1122 = s1121.v0;
        s1123 = s1121.v1;
        s1124 = svld2_f64(pg1, (X + ((2)*((j1 + ((11)*(l1)))))));
        s1125 = s1124.v0;
        s1126 = s1124.v1;
        s1127 = svld2_f64(pg1, (X + ((2)*((j1 + ((12)*(l1)))))));
        s1128 = s1127.v0;
        s1129 = s1127.v1;
        s1130 = svld2_f64(pg1, (X + ((2)*((j1 + ((13)*(l1)))))));
        s1131 = s1130.v0;
        s1132 = s1130.v1;
        s1133 = svld2_f64(pg1, (X + ((2)*((j1 + ((14)*(l1)))))));
        s1134 = s1133.v0;
        s1135 = s1133.v1;
        t2301 = svadd_f64_x(pg1, s1101, s1128);
        t2302 = svadd_f64_x(pg1, s1102, s1129);
        t2303 = svsub_f64_x(pg1, s1101, s1128);
        t2304 = svsub_f64_x(pg1, s1102, s1129);
        t2305 = svadd_f64_x(pg1, s1110, s1119);
        t2306 = svadd_f64_x(pg1, s1111, s1120);
        t2307 = svsub_f64_x(pg1, s1110, s1119);
        t2308 = svsub_f64_x(pg1, s1111, s1120);
        t2309 = svadd_f64_x(pg1, t2301, t2305);
        t2310 = svadd_f64_x(pg1, t2302, t2306);
        t2311 = svadd_f64_x(pg1, t2303, t2308);
        t2312 = svsub_f64_x(pg1, t2304, t2307);
        t2313 = svsub_f64_x(pg1, t2303, t2308);
        t2314 = svadd_f64_x(pg1, t2304, t2307);
        t2315 = svadd_f64_x(pg1, s1092, t2309);
        t2316 = svadd_f64_x(pg1, s1093, t2310);
        t2317 = svmls_n_f64_x(pg1, s1092, t2309, 0.25);
        t2318 = svmls_n_f64_x(pg1, s1093, t2310, 0.25);
        s1238 = svmla_n_f64_x(pg1, t2311, t2312, 1.6180339887498947);
        s1136 = svmul_n_f64_x(pg1, s1238, 0.29389262614623657);
        s1239 = svmls_n_f64_x(pg1, t2312, t2311, 1.6180339887498947);
        s1137 = svmul_n_f64_x(pg1, s1239, 0.29389262614623657);
        s1138 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2301, t2305), 0.55901699437494745);
        s1139 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2302, t2306), 0.55901699437494745);
        s1240 = svmls_n_f64_x(pg1, t2314, t2313, 0.61803398874989479);
        s1140 = svmul_n_f64_x(pg1, s1240, 0.47552825814757682);
        s1241 = svmla_n_f64_x(pg1, t2313, t2314, 0.61803398874989479);
        s1141 = svmul_n_f64_x(pg1, s1241, 0.47552825814757682);
        t2319 = svadd_f64_x(pg1, t2317, s1138);
        t2320 = svadd_f64_x(pg1, t2318, s1139);
        t2321 = svsub_f64_x(pg1, t2317, s1138);
        t2322 = svsub_f64_x(pg1, t2318, s1139);
        t2323 = svadd_f64_x(pg1, s1136, s1140);
        t2324 = svsub_f64_x(pg1, s1137, s1141);
        t2325 = svsub_f64_x(pg1, s1136, s1140);
        t2326 = svadd_f64_x(pg1, s1137, s1141);
        t2327 = svadd_f64_x(pg1, t2319, t2323);
        t2328 = svadd_f64_x(pg1, t2320, t2324);
        t2329 = svsub_f64_x(pg1, t2319, t2323);
        t2330 = svsub_f64_x(pg1, t2320, t2324);
        t2331 = svadd_f64_x(pg1, t2321, t2326);
        t2332 = svsub_f64_x(pg1, t2322, t2325);
        t2333 = svsub_f64_x(pg1, t2321, t2326);
        t2334 = svadd_f64_x(pg1, t2322, t2325);
        t2335 = svadd_f64_x(pg1, s1104, s1131);
        t2336 = svadd_f64_x(pg1, s1105, s1132);
        t2337 = svsub_f64_x(pg1, s1104, s1131);
        t2338 = svsub_f64_x(pg1, s1105, s1132);
        t2339 = svadd_f64_x(pg1, s1113, s1122);
        t2340 = svadd_f64_x(pg1, s1114, s1123);
        t2341 = svsub_f64_x(pg1, s1113, s1122);
        t2342 = svsub_f64_x(pg1, s1114, s1123);
        t2343 = svadd_f64_x(pg1, t2335, t2339);
        t2344 = svadd_f64_x(pg1, t2336, t2340);
        t2345 = svadd_f64_x(pg1, t2337, t2342);
        t2346 = svsub_f64_x(pg1, t2338, t2341);
        t2347 = svsub_f64_x(pg1, t2337, t2342);
        t2348 = svadd_f64_x(pg1, t2338, t2341);
        t2349 = svadd_f64_x(pg1, s1095, t2343);
        t2350 = svadd_f64_x(pg1, s1096, t2344);
        t2351 = svmls_n_f64_x(pg1, s1095, t2343, 0.25);
        t2352 = svmls_n_f64_x(pg1, s1096, t2344, 0.25);
        s1242 = svmla_n_f64_x(pg1, t2345, t2346, 1.6180339887498947);
        s1142 = svmul_n_f64_x(pg1, s1242, 0.29389262614623657);
        s1243 = svmls_n_f64_x(pg1, t2346, t2345, 1.6180339887498947);
        s1143 = svmul_n_f64_x(pg1, s1243, 0.29389262614623657);
        s1144 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2335, t2339), 0.55901699437494745);
        s1145 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2336, t2340), 0.55901699437494745);
        s1244 = svmls_n_f64_x(pg1, t2348, t2347, 0.61803398874989479);
        s1146 = svmul_n_f64_x(pg1, s1244, 0.47552825814757682);
        s1245 = svmla_n_f64_x(pg1, t2347, t2348, 0.61803398874989479);
        s1147 = svmul_n_f64_x(pg1, s1245, 0.47552825814757682);
        t2353 = svadd_f64_x(pg1, t2351, s1144);
        t2354 = svadd_f64_x(pg1, t2352, s1145);
        t2355 = svsub_f64_x(pg1, t2351, s1144);
        t2356 = svsub_f64_x(pg1, t2352, s1145);
        t2357 = svadd_f64_x(pg1, s1142, s1146);
        t2358 = svsub_f64_x(pg1, s1143, s1147);
        t2359 = svsub_f64_x(pg1, s1142, s1146);
        t2360 = svadd_f64_x(pg1, s1143, s1147);
        t2361 = svadd_f64_x(pg1, t2353, t2357);
        t2362 = svadd_f64_x(pg1, t2354, t2358);
        t2363 = svsub_f64_x(pg1, t2353, t2357);
        t2364 = svsub_f64_x(pg1, t2354, t2358);
        s1246 = svmla_n_f64_x(pg1, t2361, t2362, 0.4452286853085361);
        s1148 = svmul_n_f64_x(pg1, s1246, 0.91354545764260087);
        s1247 = svmls_n_f64_x(pg1, t2362, t2361, 0.4452286853085361);
        s1149 = svmul_n_f64_x(pg1, s1247, 0.91354545764260087);
        s1248 = svmls_n_f64_x(pg1, t2364, t2363, 0.10510423526567646);
        s1150 = svmul_n_f64_x(pg1, s1248, 0.99452189536827329);
        s1249 = svmla_n_f64_x(pg1, t2363, t2364, 0.10510423526567646);
        s1151 = svmul_n_f64_x(pg1, s1249, 0.99452189536827329);
        t2365 = svadd_f64_x(pg1, t2355, t2360);
        t2366 = svsub_f64_x(pg1, t2356, t2359);
        t2367 = svsub_f64_x(pg1, t2355, t2360);
        t2368 = svadd_f64_x(pg1, t2356, t2359);
        s1250 = svmla_n_f64_x(pg1, t2365, t2366, 1.1106125148291928);
        s1152 = svmul_n_f64_x(pg1, s1250, 0.66913060635885824);
        s1251 = svmls_n_f64_x(pg1, t2366, t2365, 1.1106125148291928);
        s1153 = svmul_n_f64_x(pg1, s1251, 0.66913060635885824);
        s1252 = svmla_n_f64_x(pg1, t2367, t2368, 3.0776835371752536);
        s1154 = svmul_n_f64_x(pg1, s1252, 0.3090169943749474);
        s1253 = svmls_n_f64_x(pg1, t2368, t2367, 3.0776835371752536);
        s1155 = svmul_n_f64_x(pg1, s1253, 0.3090169943749474);
        t2369 = svadd_f64_x(pg1, s1107, s1134);
        t2370 = svadd_f64_x(pg1, s1108, s1135);
        t2371 = svsub_f64_x(pg1, s1107, s1134);
        t2372 = svsub_f64_x(pg1, s1108, s1135);
        t2373 = svadd_f64_x(pg1, s1116, s1125);
        t2374 = svadd_f64_x(pg1, s1117, s1126);
        t2375 = svsub_f64_x(pg1, s1116, s1125);
        t2376 = svsub_f64_x(pg1, s1117, s1126);
        t2377 = svadd_f64_x(pg1, t2369, t2373);
        t2378 = svadd_f64_x(pg1, t2370, t2374);
        t2379 = svadd_f64_x(pg1, t2371, t2376);
        t2380 = svsub_f64_x(pg1, t2372, t2375);
        t2381 = svsub_f64_x(pg1, t2371, t2376);
        t2382 = svadd_f64_x(pg1, t2372, t2375);
        t2383 = svadd_f64_x(pg1, s1098, t2377);
        t2384 = svadd_f64_x(pg1, s1099, t2378);
        t2385 = svmls_n_f64_x(pg1, s1098, t2377, 0.25);
        t2386 = svmls_n_f64_x(pg1, s1099, t2378, 0.25);
        s1254 = svmla_n_f64_x(pg1, t2379, t2380, 1.6180339887498947);
        s1156 = svmul_n_f64_x(pg1, s1254, 0.29389262614623657);
        s1255 = svmls_n_f64_x(pg1, t2380, t2379, 1.6180339887498947);
        s1157 = svmul_n_f64_x(pg1, s1255, 0.29389262614623657);
        s1158 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2369, t2373), 0.55901699437494745);
        s1159 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2370, t2374), 0.55901699437494745);
        s1256 = svmls_n_f64_x(pg1, t2382, t2381, 0.61803398874989479);
        s1160 = svmul_n_f64_x(pg1, s1256, 0.47552825814757682);
        s1257 = svmla_n_f64_x(pg1, t2381, t2382, 0.61803398874989479);
        s1161 = svmul_n_f64_x(pg1, s1257, 0.47552825814757682);
        t2387 = svadd_f64_x(pg1, t2385, s1158);
        t2388 = svadd_f64_x(pg1, t2386, s1159);
        t2389 = svsub_f64_x(pg1, t2385, s1158);
        t2390 = svsub_f64_x(pg1, t2386, s1159);
        t2391 = svadd_f64_x(pg1, s1156, s1160);
        t2392 = svsub_f64_x(pg1, s1157, s1161);
        t2393 = svsub_f64_x(pg1, s1156, s1160);
        t2394 = svadd_f64_x(pg1, s1157, s1161);
        t2395 = svadd_f64_x(pg1, t2387, t2391);
        t2396 = svadd_f64_x(pg1, t2388, t2392);
        t2397 = svsub_f64_x(pg1, t2387, t2391);
        t2398 = svsub_f64_x(pg1, t2388, t2392);
        s1258 = svmla_n_f64_x(pg1, t2395, t2396, 1.1106125148291928);
        s1162 = svmul_n_f64_x(pg1, s1258, 0.66913060635885824);
        s1259 = svmls_n_f64_x(pg1, t2396, t2395, 1.1106125148291928);
        s1163 = svmul_n_f64_x(pg1, s1259, 0.66913060635885824);
        s1260 = svmla_n_f64_x(pg1, t2397, t2398, 0.2125565616700221);
        s1164 = svmul_n_f64_x(pg1, s1260, 0.97814760073380569);
        s1261 = svmls_n_f64_x(pg1, t2397, t2398, 4.7046301094784546);
        s1165 = svmul_n_f64_x(pg1, s1261, 0.20791169081775931);
        t2399 = svadd_f64_x(pg1, t2389, t2394);
        t2400 = svsub_f64_x(pg1, t2390, t2393);
        t2401 = svsub_f64_x(pg1, t2389, t2394);
        t2402 = svadd_f64_x(pg1, t2390, t2393);
        s1262 = svmls_n_f64_x(pg1, t2400, t2399, 0.10510423526567646);
        s1166 = svmul_n_f64_x(pg1, s1262, 0.99452189536827329);
        s1263 = svmla_n_f64_x(pg1, t2399, t2400, 0.10510423526567646);
        s1167 = svmul_n_f64_x(pg1, s1263, 0.99452189536827329);
        s1264 = svmls_n_f64_x(pg1, t2402, t2401, 1.3763819204711736);
        s1168 = svmul_n_f64_x(pg1, s1264, 0.58778525229247314);
        s1265 = svmla_n_f64_x(pg1, t2401, t2402, 1.3763819204711736);
        s1169 = svmul_n_f64_x(pg1, s1265, 0.58778525229247314);
        t2403 = svadd_f64_x(pg1, t2349, t2383);
        t2404 = svadd_f64_x(pg1, t2350, t2384);
        t2405 = svmls_n_f64_x(pg1, t2315, t2403, 0.5);
        t2406 = svmls_n_f64_x(pg1, t2316, t2404, 0.5);
        s1170 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2350, t2384), 0.8660254037844386);
        s1171 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t2349, t2383), 0.8660254037844386);
        s1172 = svadd_f64_x(pg1, t2315, t2403);
        s1173 = svadd_f64_x(pg1, t2316, t2404);
        s1174 = svadd_f64_x(pg1, t2405, s1170);
        s1175 = svsub_f64_x(pg1, t2406, s1171);
        s1176 = svsub_f64_x(pg1, t2405, s1170);
        s1177 = svadd_f64_x(pg1, t2406, s1171);
        t2407 = svadd_f64_x(pg1, s1148, s1162);
        t2408 = svadd_f64_x(pg1, s1149, s1163);
        t2409 = svmls_n_f64_x(pg1, t2327, t2407, 0.5);
        t2410 = svmls_n_f64_x(pg1, t2328, t2408, 0.5);
        s1178 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1149, s1163), 0.8660254037844386);
        s1179 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1148, s1162), 0.8660254037844386);
        s1180 = svadd_f64_x(pg1, t2327, t2407);
        s1181 = svadd_f64_x(pg1, t2328, t2408);
        s1182 = svadd_f64_x(pg1, t2409, s1178);
        s1183 = svsub_f64_x(pg1, t2410, s1179);
        s1184 = svsub_f64_x(pg1, t2409, s1178);
        s1185 = svadd_f64_x(pg1, t2410, s1179);
        t2411 = svadd_f64_x(pg1, s1152, s1166);
        t2412 = svsub_f64_x(pg1, s1153, s1167);
        t2413 = svmls_n_f64_x(pg1, t2331, t2411, 0.5);
        t2414 = svmls_n_f64_x(pg1, t2332, t2412, 0.5);
        s1186 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1153, s1167), 0.8660254037844386);
        s1187 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1152, s1166), 0.8660254037844386);
        s1188 = svadd_f64_x(pg1, t2331, t2411);
        s1189 = svadd_f64_x(pg1, t2332, t2412);
        s1190 = svadd_f64_x(pg1, t2413, s1186);
        s1191 = svsub_f64_x(pg1, t2414, s1187);
        s1192 = svsub_f64_x(pg1, t2413, s1186);
        s1193 = svadd_f64_x(pg1, t2414, s1187);
        t2415 = svadd_f64_x(pg1, s1154, s1168);
        t2416 = svsub_f64_x(pg1, s1155, s1169);
        t2417 = svmls_n_f64_x(pg1, t2333, t2415, 0.5);
        t2418 = svmls_n_f64_x(pg1, t2334, t2416, 0.5);
        s1194 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1155, s1169), 0.8660254037844386);
        s1195 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1154, s1168), 0.8660254037844386);
        s1196 = svadd_f64_x(pg1, t2333, t2415);
        s1197 = svadd_f64_x(pg1, t2334, t2416);
        s1198 = svadd_f64_x(pg1, t2417, s1194);
        s1199 = svsub_f64_x(pg1, t2418, s1195);
        s1200 = svsub_f64_x(pg1, t2417, s1194);
        s1201 = svadd_f64_x(pg1, t2418, s1195);
        t2419 = svsub_f64_x(pg1, s1150, s1164);
        t2420 = svsub_f64_x(pg1, s1165, s1151);
        t2421 = svmls_n_f64_x(pg1, t2329, t2419, 0.5);
        t2422 = svmls_n_f64_x(pg1, t2330, t2420, 0.5);
        s1202 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1151, s1165), 0.8660254037844386);
        s1203 = svmul_n_f64_x(pg1, svadd_f64_x(pg1, s1150, s1164), 0.8660254037844386);
        s1204 = svadd_f64_x(pg1, t2329, t2419);
        s1205 = svadd_f64_x(pg1, t2330, t2420);
        s1206 = svsub_f64_x(pg1, t2421, s1202);
        s1207 = svsub_f64_x(pg1, t2422, s1203);
        s1208 = svadd_f64_x(pg1, t2421, s1202);
        s1209 = svadd_f64_x(pg1, t2422, s1203);
        a1724 = (TW1 + ((28)*(j1)));
        a1725 = svld1_gather_u64offset_f64(pg1, a1724, svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1726 = svld1_gather_u64offset_f64(pg1, (a1724 + 1), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1210 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1726, s1181), a1725, s1180);
        s1211 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1725, s1181), a1726, s1180);
        a1727 = svld1_gather_u64offset_f64(pg1, (a1724 + 2), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1728 = svld1_gather_u64offset_f64(pg1, (a1724 + 3), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1212 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1728, s1189), a1727, s1188);
        s1213 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1727, s1189), a1728, s1188);
        a1729 = svld1_gather_u64offset_f64(pg1, (a1724 + 4), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1730 = svld1_gather_u64offset_f64(pg1, (a1724 + 5), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1214 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1730, s1197), a1729, s1196);
        s1215 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1729, s1197), a1730, s1196);
        a1731 = svld1_gather_u64offset_f64(pg1, (a1724 + 6), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1732 = svld1_gather_u64offset_f64(pg1, (a1724 + 7), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1216 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1732, s1205), a1731, s1204);
        s1217 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1731, s1205), a1732, s1204);
        a1733 = svld1_gather_u64offset_f64(pg1, (a1724 + 8), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1734 = svld1_gather_u64offset_f64(pg1, (a1724 + 9), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1218 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1734, s1175), a1733, s1174);
        s1219 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1733, s1175), a1734, s1174);
        a1735 = svld1_gather_u64offset_f64(pg1, (a1724 + 10), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1736 = svld1_gather_u64offset_f64(pg1, (a1724 + 11), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1220 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1736, s1183), a1735, s1182);
        s1221 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1735, s1183), a1736, s1182);
        a1737 = svld1_gather_u64offset_f64(pg1, (a1724 + 12), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1738 = svld1_gather_u64offset_f64(pg1, (a1724 + 13), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1222 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1738, s1191), a1737, s1190);
        s1223 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1737, s1191), a1738, s1190);
        a1739 = svld1_gather_u64offset_f64(pg1, (a1724 + 14), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1740 = svld1_gather_u64offset_f64(pg1, (a1724 + 15), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1224 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1740, s1199), a1739, s1198);
        s1225 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1739, s1199), a1740, s1198);
        a1741 = svld1_gather_u64offset_f64(pg1, (a1724 + 16), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1742 = svld1_gather_u64offset_f64(pg1, (a1724 + 17), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1226 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1742, s1207), a1741, s1206);
        s1227 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1741, s1207), a1742, s1206);
        a1743 = svld1_gather_u64offset_f64(pg1, (a1724 + 18), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1744 = svld1_gather_u64offset_f64(pg1, (a1724 + 19), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1228 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1744, s1177), a1743, s1176);
        s1229 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1743, s1177), a1744, s1176);
        a1745 = svld1_gather_u64offset_f64(pg1, (a1724 + 20), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1746 = svld1_gather_u64offset_f64(pg1, (a1724 + 21), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1230 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1746, s1185), a1745, s1184);
        s1231 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1745, s1185), a1746, s1184);
        a1747 = svld1_gather_u64offset_f64(pg1, (a1724 + 22), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1748 = svld1_gather_u64offset_f64(pg1, (a1724 + 23), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1232 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1748, s1193), a1747, s1192);
        s1233 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1747, s1193), a1748, s1192);
        a1749 = svld1_gather_u64offset_f64(pg1, (a1724 + 24), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1750 = svld1_gather_u64offset_f64(pg1, (a1724 + 25), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1234 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1750, s1201), a1749, s1200);
        s1235 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1749, s1201), a1750, s1200);
        a1751 = svld1_gather_u64offset_f64(pg1, (a1724 + 26), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        a1752 = svld1_gather_u64offset_f64(pg1, (a1724 + 27), svindex_u64(0, (int64_t)(28 * sizeof(float64_t))));
        s1236 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1752, s1209), a1751, s1208);
        s1237 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1751, s1209), a1752, s1208);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1172);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1173);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1210);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1211);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1212);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1213);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1214);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1215);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(8 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1216);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(9 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1217);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(10 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1218);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(11 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1219);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(12 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1220);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(13 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1221);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(14 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1222);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(15 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1223);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(16 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1224);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(17 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1225);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(18 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1226);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(19 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1227);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(20 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1228);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(21 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1229);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(22 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1230);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(23 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1231);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(24 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1232);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(25 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1233);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(26 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1234);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(27 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1235);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(28 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1236);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(29 + Y + ((30)*(j1)))), svindex_u64(0, (int64_t)(30 * sizeof(float64_t))), s1237);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}
