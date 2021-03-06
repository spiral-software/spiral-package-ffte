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

void dft10a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    int l1;
    float64_t  *a1022;
    svfloat64x2_t s586, s589, s592, s595, s598, s601, s604, s607, 
            s610, s613;
    svfloat64_t a1023, a1024, a1025, a1026, a1027, a1028, a1029, a1030, 
            a1031, a1032, a1033, a1034, a1035, a1036, a1037, a1038, 
            a1039, a1040, s587, s588, s590, s591, s593, s594, 
            s596, s597, s599, s600, s602, s603, s605, s606, 
            s608, s609, s611, s612, s614, s615, s616, s617, 
            s618, s619, s620, s621, s622, s623, s624, s625, 
            s626, s627, s628, s629, s630, s631, s632, s633, 
            s634, s635, s636, s637, s638, s639, s640, s641, 
            s642, s643, s644, s645, s646, s647, s648, s649, 
            s650, s651, s652, s653, s654, s655, s656, s657, 
            s658, s659, s660, s661, s662, s663, s664, s665, 
            s666, s667, s668, s669, s670, s671, s672, s673, 
            s674, s675, s676, s677, s678, s679, s680, s681, 
            s682, s683, s684, s685, s686, s687, s688, s689, 
            t1311, t1312, t1313, t1314, t1315, t1316, t1317, t1318, 
            t1319, t1320, t1321, t1322, t1323, t1324, t1325, t1326, 
            t1327, t1328, t1329, t1330, t1331, t1332, t1333, t1334, 
            t1335, t1336, t1337, t1338, t1339, t1340, t1341, t1342, 
            t1343, t1344, t1345, t1346, t1347, t1348, t1349, t1350, 
            t1351, t1352, t1353, t1354, t1355, t1356, t1357, t1358, 
            t1359, t1360, t1361, t1362, t1363, t1364, t1365, t1366, 
            t1367, t1368, t1369, t1370, t1371, t1372, t1373, t1374, 
            t1375, t1376, t1377, t1378;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s586 = svld2_f64(pg1, (X + ((2)*(j1))));
        s587 = s586.v0;
        s588 = s586.v1;
        s589 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s590 = s589.v0;
        s591 = s589.v1;
        s592 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s593 = s592.v0;
        s594 = s592.v1;
        s595 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s596 = s595.v0;
        s597 = s595.v1;
        s598 = svld2_f64(pg1, (X + ((2)*((j1 + ((4)*(l1)))))));
        s599 = s598.v0;
        s600 = s598.v1;
        s601 = svld2_f64(pg1, (X + ((2)*((j1 + ((5)*(l1)))))));
        s602 = s601.v0;
        s603 = s601.v1;
        s604 = svld2_f64(pg1, (X + ((2)*((j1 + ((6)*(l1)))))));
        s605 = s604.v0;
        s606 = s604.v1;
        s607 = svld2_f64(pg1, (X + ((2)*((j1 + ((7)*(l1)))))));
        s608 = s607.v0;
        s609 = s607.v1;
        s610 = svld2_f64(pg1, (X + ((2)*((j1 + ((8)*(l1)))))));
        s611 = s610.v0;
        s612 = s610.v1;
        s613 = svld2_f64(pg1, (X + ((2)*((j1 + ((9)*(l1)))))));
        s614 = s613.v0;
        s615 = s613.v1;
        t1311 = svadd_f64_x(pg1, s593, s611);
        t1312 = svadd_f64_x(pg1, s594, s612);
        t1313 = svsub_f64_x(pg1, s593, s611);
        t1314 = svsub_f64_x(pg1, s594, s612);
        t1315 = svadd_f64_x(pg1, s599, s605);
        t1316 = svadd_f64_x(pg1, s600, s606);
        t1317 = svsub_f64_x(pg1, s599, s605);
        t1318 = svsub_f64_x(pg1, s600, s606);
        t1319 = svadd_f64_x(pg1, t1311, t1315);
        t1320 = svadd_f64_x(pg1, t1312, t1316);
        t1321 = svadd_f64_x(pg1, t1313, t1318);
        t1322 = svsub_f64_x(pg1, t1314, t1317);
        t1323 = svsub_f64_x(pg1, t1313, t1318);
        t1324 = svadd_f64_x(pg1, t1314, t1317);
        t1325 = svadd_f64_x(pg1, s587, t1319);
        t1326 = svadd_f64_x(pg1, s588, t1320);
        t1327 = svmls_n_f64_x(pg1, s587, t1319, 0.25);
        t1328 = svmls_n_f64_x(pg1, s588, t1320, 0.25);
        s674 = svmla_n_f64_x(pg1, t1321, t1322, 1.6180339887498947);
        s616 = svmul_n_f64_x(pg1, s674, 0.29389262614623657);
        s675 = svmls_n_f64_x(pg1, t1322, t1321, 1.6180339887498947);
        s617 = svmul_n_f64_x(pg1, s675, 0.29389262614623657);
        s618 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1311, t1315), 0.55901699437494745);
        s619 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1312, t1316), 0.55901699437494745);
        s676 = svmls_n_f64_x(pg1, t1324, t1323, 0.61803398874989479);
        s620 = svmul_n_f64_x(pg1, s676, 0.47552825814757682);
        s677 = svmla_n_f64_x(pg1, t1323, t1324, 0.61803398874989479);
        s621 = svmul_n_f64_x(pg1, s677, 0.47552825814757682);
        t1329 = svadd_f64_x(pg1, t1327, s618);
        t1330 = svadd_f64_x(pg1, t1328, s619);
        t1331 = svsub_f64_x(pg1, t1327, s618);
        t1332 = svsub_f64_x(pg1, t1328, s619);
        t1333 = svadd_f64_x(pg1, s616, s620);
        t1334 = svsub_f64_x(pg1, s617, s621);
        t1335 = svsub_f64_x(pg1, s616, s620);
        t1336 = svadd_f64_x(pg1, s617, s621);
        t1337 = svadd_f64_x(pg1, t1329, t1333);
        t1338 = svadd_f64_x(pg1, t1330, t1334);
        t1339 = svsub_f64_x(pg1, t1329, t1333);
        t1340 = svsub_f64_x(pg1, t1330, t1334);
        t1341 = svadd_f64_x(pg1, t1331, t1336);
        t1342 = svsub_f64_x(pg1, t1332, t1335);
        t1343 = svsub_f64_x(pg1, t1331, t1336);
        t1344 = svadd_f64_x(pg1, t1332, t1335);
        t1345 = svadd_f64_x(pg1, s596, s614);
        t1346 = svadd_f64_x(pg1, s597, s615);
        t1347 = svsub_f64_x(pg1, s596, s614);
        t1348 = svsub_f64_x(pg1, s597, s615);
        t1349 = svadd_f64_x(pg1, s602, s608);
        t1350 = svadd_f64_x(pg1, s603, s609);
        t1351 = svsub_f64_x(pg1, s602, s608);
        t1352 = svsub_f64_x(pg1, s603, s609);
        t1353 = svadd_f64_x(pg1, t1345, t1349);
        t1354 = svadd_f64_x(pg1, t1346, t1350);
        t1355 = svadd_f64_x(pg1, t1347, t1352);
        t1356 = svsub_f64_x(pg1, t1348, t1351);
        t1357 = svsub_f64_x(pg1, t1347, t1352);
        t1358 = svadd_f64_x(pg1, t1348, t1351);
        t1359 = svadd_f64_x(pg1, s590, t1353);
        t1360 = svadd_f64_x(pg1, s591, t1354);
        t1361 = svmls_n_f64_x(pg1, s590, t1353, 0.25);
        t1362 = svmls_n_f64_x(pg1, s591, t1354, 0.25);
        s678 = svmla_n_f64_x(pg1, t1355, t1356, 1.6180339887498947);
        s622 = svmul_n_f64_x(pg1, s678, 0.29389262614623657);
        s679 = svmls_n_f64_x(pg1, t1356, t1355, 1.6180339887498947);
        s623 = svmul_n_f64_x(pg1, s679, 0.29389262614623657);
        s624 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1345, t1349), 0.55901699437494745);
        s625 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1346, t1350), 0.55901699437494745);
        s680 = svmls_n_f64_x(pg1, t1358, t1357, 0.61803398874989479);
        s626 = svmul_n_f64_x(pg1, s680, 0.47552825814757682);
        s681 = svmla_n_f64_x(pg1, t1357, t1358, 0.61803398874989479);
        s627 = svmul_n_f64_x(pg1, s681, 0.47552825814757682);
        t1363 = svadd_f64_x(pg1, t1361, s624);
        t1364 = svadd_f64_x(pg1, t1362, s625);
        t1365 = svsub_f64_x(pg1, t1361, s624);
        t1366 = svsub_f64_x(pg1, t1362, s625);
        t1367 = svadd_f64_x(pg1, s622, s626);
        t1368 = svsub_f64_x(pg1, s623, s627);
        t1369 = svsub_f64_x(pg1, s622, s626);
        t1370 = svadd_f64_x(pg1, s623, s627);
        t1371 = svadd_f64_x(pg1, t1363, t1367);
        t1372 = svadd_f64_x(pg1, t1364, t1368);
        t1373 = svsub_f64_x(pg1, t1363, t1367);
        t1374 = svsub_f64_x(pg1, t1364, t1368);
        s682 = svmla_n_f64_x(pg1, t1371, t1372, 0.7265425280053609);
        s628 = svmul_n_f64_x(pg1, s682, 0.80901699437494745);
        s683 = svmls_n_f64_x(pg1, t1372, t1371, 0.7265425280053609);
        s629 = svmul_n_f64_x(pg1, s683, 0.80901699437494745);
        s684 = svmls_n_f64_x(pg1, t1374, t1373, 1.3763819204711736);
        s630 = svmul_n_f64_x(pg1, s684, 0.58778525229247314);
        s685 = svmla_n_f64_x(pg1, t1373, t1374, 1.3763819204711736);
        s631 = svmul_n_f64_x(pg1, s685, 0.58778525229247314);
        t1375 = svadd_f64_x(pg1, t1365, t1370);
        t1376 = svsub_f64_x(pg1, t1366, t1369);
        t1377 = svsub_f64_x(pg1, t1365, t1370);
        t1378 = svadd_f64_x(pg1, t1366, t1369);
        s686 = svmla_n_f64_x(pg1, t1375, t1376, 3.0776835371752536);
        s632 = svmul_n_f64_x(pg1, s686, 0.3090169943749474);
        s687 = svmls_n_f64_x(pg1, t1376, t1375, 3.0776835371752536);
        s633 = svmul_n_f64_x(pg1, s687, 0.3090169943749474);
        s688 = svmls_n_f64_x(pg1, t1378, t1377, 0.32491969623290629);
        s634 = svmul_n_f64_x(pg1, s688, 0.95105651629515353);
        s689 = svmla_n_f64_x(pg1, t1377, t1378, 0.32491969623290629);
        s635 = svmul_n_f64_x(pg1, s689, 0.95105651629515353);
        s636 = svadd_f64_x(pg1, t1325, t1359);
        s637 = svadd_f64_x(pg1, t1326, t1360);
        s638 = svsub_f64_x(pg1, t1325, t1359);
        s639 = svsub_f64_x(pg1, t1326, t1360);
        s640 = svadd_f64_x(pg1, t1337, s628);
        s641 = svadd_f64_x(pg1, t1338, s629);
        s642 = svsub_f64_x(pg1, t1337, s628);
        s643 = svsub_f64_x(pg1, t1338, s629);
        s644 = svadd_f64_x(pg1, t1341, s632);
        s645 = svadd_f64_x(pg1, t1342, s633);
        s646 = svsub_f64_x(pg1, t1341, s632);
        s647 = svsub_f64_x(pg1, t1342, s633);
        s648 = svadd_f64_x(pg1, t1343, s634);
        s649 = svsub_f64_x(pg1, t1344, s635);
        s650 = svsub_f64_x(pg1, t1343, s634);
        s651 = svadd_f64_x(pg1, t1344, s635);
        s652 = svadd_f64_x(pg1, t1339, s630);
        s653 = svsub_f64_x(pg1, t1340, s631);
        s654 = svsub_f64_x(pg1, t1339, s630);
        s655 = svadd_f64_x(pg1, t1340, s631);
        a1022 = (TW1 + ((18)*(j1)));
        a1023 = svld1_gather_u64offset_f64(pg1, a1022, svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1024 = svld1_gather_u64offset_f64(pg1, (a1022 + 1), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s656 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1024, s641), a1023, s640);
        s657 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1023, s641), a1024, s640);
        a1025 = svld1_gather_u64offset_f64(pg1, (a1022 + 2), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1026 = svld1_gather_u64offset_f64(pg1, (a1022 + 3), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s658 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1026, s645), a1025, s644);
        s659 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1025, s645), a1026, s644);
        a1027 = svld1_gather_u64offset_f64(pg1, (a1022 + 4), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1028 = svld1_gather_u64offset_f64(pg1, (a1022 + 5), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s660 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1028, s649), a1027, s648);
        s661 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1027, s649), a1028, s648);
        a1029 = svld1_gather_u64offset_f64(pg1, (a1022 + 6), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1030 = svld1_gather_u64offset_f64(pg1, (a1022 + 7), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s662 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1030, s653), a1029, s652);
        s663 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1029, s653), a1030, s652);
        a1031 = svld1_gather_u64offset_f64(pg1, (a1022 + 8), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1032 = svld1_gather_u64offset_f64(pg1, (a1022 + 9), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s664 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1032, s639), a1031, s638);
        s665 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1031, s639), a1032, s638);
        a1033 = svld1_gather_u64offset_f64(pg1, (a1022 + 10), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1034 = svld1_gather_u64offset_f64(pg1, (a1022 + 11), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s666 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1034, s643), a1033, s642);
        s667 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1033, s643), a1034, s642);
        a1035 = svld1_gather_u64offset_f64(pg1, (a1022 + 12), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1036 = svld1_gather_u64offset_f64(pg1, (a1022 + 13), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s668 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1036, s647), a1035, s646);
        s669 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1035, s647), a1036, s646);
        a1037 = svld1_gather_u64offset_f64(pg1, (a1022 + 14), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1038 = svld1_gather_u64offset_f64(pg1, (a1022 + 15), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s670 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1038, s651), a1037, s650);
        s671 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1037, s651), a1038, s650);
        a1039 = svld1_gather_u64offset_f64(pg1, (a1022 + 16), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        a1040 = svld1_gather_u64offset_f64(pg1, (a1022 + 17), svindex_u64(0, (int64_t)(18 * sizeof(float64_t))));
        s672 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a1040, s655), a1039, s654);
        s673 = svmla_f64_x(pg1, svmul_f64_x(pg1, a1039, s655), a1040, s654);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s636);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s637);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s656);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s657);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s658);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s659);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s660);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s661);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(8 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s662);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(9 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s663);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(10 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s664);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(11 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s665);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(12 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s666);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(13 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s667);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(14 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s668);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(15 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s669);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(16 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s670);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(17 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s671);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(18 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s672);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(19 + Y + ((20)*(j1)))), svindex_u64(0, (int64_t)(20 * sizeof(float64_t))), s673);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}
