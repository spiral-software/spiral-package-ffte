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

void dft10b_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1, int  *mp1) {
    float64_t a1075, a1076, a1077, a1078, a1079, a1080, a1081, a1082, 
            a1083, a1084, a1085, a1086, a1087, a1088, a1089, a1090, 
            a1091, a1092;
    int a1073, a1074, j1, l1, m1;
    svfloat64x2_t s566, s569, s572, s575, s578, s581, s584, s587, 
            s590, s593, svex2_11, svex2_12, svex2_13, svex2_14, svex2_15, svex2_16, 
            svex2_17, svex2_18, svex2_19, svex2_20;
    svfloat64_t s567, s568, s570, s571, s573, s574, s576, s577, 
            s579, s580, s582, s583, s585, s586, s588, s589, 
            s591, s592, s594, s595, s596, s597, s598, s599, 
            s600, s601, s602, s603, s604, s605, s606, s607, 
            s608, s609, s610, s611, s612, s613, s614, s615, 
            s616, s617, s618, s619, s620, s621, s622, s623, 
            s624, s625, s626, s627, s628, s629, s630, s631, 
            s632, s633, s634, s635, s636, s637, s638, s639, 
            s640, s641, s642, s643, s644, s645, s646, s647, 
            s648, s649, t1311, t1312, t1313, t1314, t1315, t1316, 
            t1317, t1318, t1319, t1320, t1321, t1322, t1323, t1324, 
            t1325, t1326, t1327, t1328, t1329, t1330, t1331, t1332, 
            t1333, t1334, t1335, t1336, t1337, t1338, t1339, t1340, 
            t1341, t1342, t1343, t1344, t1345, t1346, t1347, t1348, 
            t1349, t1350, t1351, t1352, t1353, t1354, t1355, t1356, 
            t1357, t1358, t1359, t1360, t1361, t1362, t1363, t1364, 
            t1365, t1366, t1367, t1368, t1369, t1370, t1371, t1372, 
            t1373, t1374, t1375, t1376, t1377, t1378;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        int k1 = 0;
        pg1 = svwhilelt_b64(k1, m1);
        do {
            a1073 = (k1 + ((j1)*(m1)));
            s566 = svld2_f64(pg1, (X + ((2)*(a1073))));
            s567 = s566.v0;
            s568 = s566.v1;
            s569 = svld2_f64(pg1, (X + ((2)*((a1073 + ((l1)*(m1)))))));
            s570 = s569.v0;
            s571 = s569.v1;
            s572 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((2)*(l1)))*(m1)))))));
            s573 = s572.v0;
            s574 = s572.v1;
            s575 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((3)*(l1)))*(m1)))))));
            s576 = s575.v0;
            s577 = s575.v1;
            s578 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((4)*(l1)))*(m1)))))));
            s579 = s578.v0;
            s580 = s578.v1;
            s581 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((5)*(l1)))*(m1)))))));
            s582 = s581.v0;
            s583 = s581.v1;
            s584 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((6)*(l1)))*(m1)))))));
            s585 = s584.v0;
            s586 = s584.v1;
            s587 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((7)*(l1)))*(m1)))))));
            s588 = s587.v0;
            s589 = s587.v1;
            s590 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((8)*(l1)))*(m1)))))));
            s591 = s590.v0;
            s592 = s590.v1;
            s593 = svld2_f64(pg1, (X + ((2)*((a1073 + ((((9)*(l1)))*(m1)))))));
            s594 = s593.v0;
            s595 = s593.v1;
            t1311 = svadd_f64_x(pg1, s573, s591);
            t1312 = svadd_f64_x(pg1, s574, s592);
            t1313 = svsub_f64_x(pg1, s573, s591);
            t1314 = svsub_f64_x(pg1, s574, s592);
            t1315 = svadd_f64_x(pg1, s579, s585);
            t1316 = svadd_f64_x(pg1, s580, s586);
            t1317 = svsub_f64_x(pg1, s579, s585);
            t1318 = svsub_f64_x(pg1, s580, s586);
            t1319 = svadd_f64_x(pg1, t1311, t1315);
            t1320 = svadd_f64_x(pg1, t1312, t1316);
            t1321 = svadd_f64_x(pg1, t1313, t1318);
            t1322 = svsub_f64_x(pg1, t1314, t1317);
            t1323 = svsub_f64_x(pg1, t1313, t1318);
            t1324 = svadd_f64_x(pg1, t1314, t1317);
            t1325 = svadd_f64_x(pg1, s567, t1319);
            t1326 = svadd_f64_x(pg1, s568, t1320);
            t1327 = svmls_n_f64_x(pg1, s567, t1319, 0.25);
            t1328 = svmls_n_f64_x(pg1, s568, t1320, 0.25);
            s634 = svmla_n_f64_x(pg1, t1321, t1322, 1.6180339887498947);
            s596 = svmul_n_f64_x(pg1, s634, 0.29389262614623657);
            s635 = svmls_n_f64_x(pg1, t1322, t1321, 1.6180339887498947);
            s597 = svmul_n_f64_x(pg1, s635, 0.29389262614623657);
            s598 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1311, t1315), 0.55901699437494745);
            s599 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1312, t1316), 0.55901699437494745);
            s636 = svmls_n_f64_x(pg1, t1324, t1323, 0.61803398874989479);
            s600 = svmul_n_f64_x(pg1, s636, 0.47552825814757682);
            s637 = svmla_n_f64_x(pg1, t1323, t1324, 0.61803398874989479);
            s601 = svmul_n_f64_x(pg1, s637, 0.47552825814757682);
            t1329 = svadd_f64_x(pg1, t1327, s598);
            t1330 = svadd_f64_x(pg1, t1328, s599);
            t1331 = svsub_f64_x(pg1, t1327, s598);
            t1332 = svsub_f64_x(pg1, t1328, s599);
            t1333 = svadd_f64_x(pg1, s596, s600);
            t1334 = svsub_f64_x(pg1, s597, s601);
            t1335 = svsub_f64_x(pg1, s596, s600);
            t1336 = svadd_f64_x(pg1, s597, s601);
            t1337 = svadd_f64_x(pg1, t1329, t1333);
            t1338 = svadd_f64_x(pg1, t1330, t1334);
            t1339 = svsub_f64_x(pg1, t1329, t1333);
            t1340 = svsub_f64_x(pg1, t1330, t1334);
            t1341 = svadd_f64_x(pg1, t1331, t1336);
            t1342 = svsub_f64_x(pg1, t1332, t1335);
            t1343 = svsub_f64_x(pg1, t1331, t1336);
            t1344 = svadd_f64_x(pg1, t1332, t1335);
            t1345 = svadd_f64_x(pg1, s576, s594);
            t1346 = svadd_f64_x(pg1, s577, s595);
            t1347 = svsub_f64_x(pg1, s576, s594);
            t1348 = svsub_f64_x(pg1, s577, s595);
            t1349 = svadd_f64_x(pg1, s582, s588);
            t1350 = svadd_f64_x(pg1, s583, s589);
            t1351 = svsub_f64_x(pg1, s582, s588);
            t1352 = svsub_f64_x(pg1, s583, s589);
            t1353 = svadd_f64_x(pg1, t1345, t1349);
            t1354 = svadd_f64_x(pg1, t1346, t1350);
            t1355 = svadd_f64_x(pg1, t1347, t1352);
            t1356 = svsub_f64_x(pg1, t1348, t1351);
            t1357 = svsub_f64_x(pg1, t1347, t1352);
            t1358 = svadd_f64_x(pg1, t1348, t1351);
            t1359 = svadd_f64_x(pg1, s570, t1353);
            t1360 = svadd_f64_x(pg1, s571, t1354);
            t1361 = svmls_n_f64_x(pg1, s570, t1353, 0.25);
            t1362 = svmls_n_f64_x(pg1, s571, t1354, 0.25);
            s638 = svmla_n_f64_x(pg1, t1355, t1356, 1.6180339887498947);
            s602 = svmul_n_f64_x(pg1, s638, 0.29389262614623657);
            s639 = svmls_n_f64_x(pg1, t1356, t1355, 1.6180339887498947);
            s603 = svmul_n_f64_x(pg1, s639, 0.29389262614623657);
            s604 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1345, t1349), 0.55901699437494745);
            s605 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1346, t1350), 0.55901699437494745);
            s640 = svmls_n_f64_x(pg1, t1358, t1357, 0.61803398874989479);
            s606 = svmul_n_f64_x(pg1, s640, 0.47552825814757682);
            s641 = svmla_n_f64_x(pg1, t1357, t1358, 0.61803398874989479);
            s607 = svmul_n_f64_x(pg1, s641, 0.47552825814757682);
            t1363 = svadd_f64_x(pg1, t1361, s604);
            t1364 = svadd_f64_x(pg1, t1362, s605);
            t1365 = svsub_f64_x(pg1, t1361, s604);
            t1366 = svsub_f64_x(pg1, t1362, s605);
            t1367 = svadd_f64_x(pg1, s602, s606);
            t1368 = svsub_f64_x(pg1, s603, s607);
            t1369 = svsub_f64_x(pg1, s602, s606);
            t1370 = svadd_f64_x(pg1, s603, s607);
            t1371 = svadd_f64_x(pg1, t1363, t1367);
            t1372 = svadd_f64_x(pg1, t1364, t1368);
            t1373 = svsub_f64_x(pg1, t1363, t1367);
            t1374 = svsub_f64_x(pg1, t1364, t1368);
            s642 = svmla_n_f64_x(pg1, t1371, t1372, 0.7265425280053609);
            s608 = svmul_n_f64_x(pg1, s642, 0.80901699437494745);
            s643 = svmls_n_f64_x(pg1, t1372, t1371, 0.7265425280053609);
            s609 = svmul_n_f64_x(pg1, s643, 0.80901699437494745);
            s644 = svmls_n_f64_x(pg1, t1374, t1373, 1.3763819204711736);
            s610 = svmul_n_f64_x(pg1, s644, 0.58778525229247314);
            s645 = svmla_n_f64_x(pg1, t1373, t1374, 1.3763819204711736);
            s611 = svmul_n_f64_x(pg1, s645, 0.58778525229247314);
            t1375 = svadd_f64_x(pg1, t1365, t1370);
            t1376 = svsub_f64_x(pg1, t1366, t1369);
            t1377 = svsub_f64_x(pg1, t1365, t1370);
            t1378 = svadd_f64_x(pg1, t1366, t1369);
            s646 = svmla_n_f64_x(pg1, t1375, t1376, 3.0776835371752536);
            s612 = svmul_n_f64_x(pg1, s646, 0.3090169943749474);
            s647 = svmls_n_f64_x(pg1, t1376, t1375, 3.0776835371752536);
            s613 = svmul_n_f64_x(pg1, s647, 0.3090169943749474);
            s648 = svmls_n_f64_x(pg1, t1378, t1377, 0.32491969623290629);
            s614 = svmul_n_f64_x(pg1, s648, 0.95105651629515353);
            s649 = svmla_n_f64_x(pg1, t1377, t1378, 0.32491969623290629);
            s615 = svmul_n_f64_x(pg1, s649, 0.95105651629515353);
            s616 = svsub_f64_x(pg1, t1325, t1359);
            s617 = svsub_f64_x(pg1, t1326, t1360);
            s618 = svadd_f64_x(pg1, t1337, s608);
            s619 = svadd_f64_x(pg1, t1338, s609);
            s620 = svsub_f64_x(pg1, t1337, s608);
            s621 = svsub_f64_x(pg1, t1338, s609);
            s622 = svadd_f64_x(pg1, t1341, s612);
            s623 = svadd_f64_x(pg1, t1342, s613);
            s624 = svsub_f64_x(pg1, t1341, s612);
            s625 = svsub_f64_x(pg1, t1342, s613);
            s626 = svadd_f64_x(pg1, t1343, s614);
            s627 = svsub_f64_x(pg1, t1344, s615);
            s628 = svsub_f64_x(pg1, t1343, s614);
            s629 = svadd_f64_x(pg1, t1344, s615);
            s630 = svadd_f64_x(pg1, t1339, s610);
            s631 = svsub_f64_x(pg1, t1340, s611);
            s632 = svsub_f64_x(pg1, t1339, s610);
            s633 = svadd_f64_x(pg1, t1340, s611);
            a1074 = ((18)*(j1));
            a1075 = TW1[a1074];
            a1076 = TW1[(a1074 + 1)];
            a1077 = TW1[(a1074 + 2)];
            a1078 = TW1[(a1074 + 3)];
            a1079 = TW1[(a1074 + 4)];
            a1080 = TW1[(a1074 + 5)];
            a1081 = TW1[(a1074 + 6)];
            a1082 = TW1[(a1074 + 7)];
            a1083 = TW1[(a1074 + 8)];
            a1084 = TW1[(a1074 + 9)];
            a1085 = TW1[(a1074 + 10)];
            a1086 = TW1[(a1074 + 11)];
            a1087 = TW1[(a1074 + 12)];
            a1088 = TW1[(a1074 + 13)];
            a1089 = TW1[(a1074 + 14)];
            a1090 = TW1[(a1074 + 15)];
            a1091 = TW1[(a1074 + 16)];
            a1092 = TW1[(a1074 + 17)];
            svex2_11.v0 = svadd_f64_x(pg1, t1325, t1359);
            svex2_11.v1 = svadd_f64_x(pg1, t1326, t1360);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)))))), svex2_11);
            svex2_12.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s619, a1076), s618, a1075);
            svex2_12.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s619, a1075), s618, a1076);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + m1)))), svex2_12);
            svex2_13.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s623, a1078), s622, a1077);
            svex2_13.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s623, a1077), s622, a1078);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((2)*(m1)))))), svex2_13);
            svex2_14.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s627, a1080), s626, a1079);
            svex2_14.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s627, a1079), s626, a1080);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((3)*(m1)))))), svex2_14);
            svex2_15.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s631, a1082), s630, a1081);
            svex2_15.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s631, a1081), s630, a1082);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((4)*(m1)))))), svex2_15);
            svex2_16.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s617, a1084), s616, a1083);
            svex2_16.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s617, a1083), s616, a1084);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((5)*(m1)))))), svex2_16);
            svex2_17.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s621, a1086), s620, a1085);
            svex2_17.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s621, a1085), s620, a1086);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((6)*(m1)))))), svex2_17);
            svex2_18.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s625, a1088), s624, a1087);
            svex2_18.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s625, a1087), s624, a1088);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((7)*(m1)))))), svex2_18);
            svex2_19.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s629, a1090), s628, a1089);
            svex2_19.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s629, a1089), s628, a1090);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((8)*(m1)))))), svex2_19);
            svex2_20.v0 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, s633, a1092), s632, a1091);
            svex2_20.v1 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, s633, a1091), s632, a1092);
            svst2_f64(pg1, (Y + ((2)*((k1 + ((((10)*(j1)))*(m1)) + ((9)*(m1)))))), svex2_20);
            k1 += svcntd();
            pg1 = svwhilelt_b64(k1, m1);
        } while(svptest_any(svptrue_b64(), pg1));
    }
}