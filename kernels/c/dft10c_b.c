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


void dft10b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    double a1293, a1294, a1295, a1296, a1297, a1298, a1299, a1300, 
            a1301, a1302, a1303, a1304, a1305, a1306, a1307, a1308, 
            a1309, a1310, s456, s457, s458, s459, s460, s461, 
            s462, s463, s464, s465, s466, s467, s468, s469, 
            s470, s471, s472, s473, s474, s475, s476, s477, 
            s478, s479, s480, s481, s482, s483, s484, s485, 
            s486, s487, s488, s489, s490, s491, s492, s493, 
            s494, s495, s496, s497, s498, s499, s500, s501, 
            s502, s503, s504, s505, s506, s507, s508, s509, 
            s510, s511, s512, s513, t1310, t1311, t1312, t1313, 
            t1314, t1315, t1316, t1317, t1318, t1319, t1320, t1321, 
            t1322, t1323, t1324, t1325, t1326, t1327, t1328, t1329, 
            t1330, t1331, t1332, t1333, t1334, t1335, t1336, t1337, 
            t1338, t1339, t1340, t1341, t1342, t1343, t1344, t1345, 
            t1346, t1347, t1348, t1349, t1350, t1351, t1352, t1353, 
            t1354, t1355, t1356, t1357, t1358, t1359, t1360, t1361, 
            t1362, t1363, t1364, t1365, t1366, t1367, t1368, t1369, 
            t1370, t1371, t1372, t1373, t1374, t1375, t1376, t1377;
    int a1280, a1281, a1282, a1283, a1284, a1285, a1286, a1287, 
            a1288, a1289, a1290, a1291, a1292, a1311, a1312, a1313, 
            a1314, a1315, a1316, a1317, a1318, a1319, a1320, a1321, 
            b150, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a1280 = (2*k1);
            a1281 = (j1*m1);
            a1282 = (a1280 + (2*a1281));
            s456 = X[a1282];
            s457 = X[(a1282 + 1)];
            b150 = (l1*m1);
            a1283 = (a1282 + (2*b150));
            s458 = X[a1283];
            s459 = X[(a1283 + 1)];
            a1284 = (a1282 + (4*b150));
            s460 = X[a1284];
            s461 = X[(a1284 + 1)];
            a1285 = (a1282 + (6*b150));
            s462 = X[a1285];
            s463 = X[(a1285 + 1)];
            a1286 = (a1282 + (8*b150));
            s464 = X[a1286];
            s465 = X[(a1286 + 1)];
            a1287 = (a1282 + (10*b150));
            s466 = X[a1287];
            s467 = X[(a1287 + 1)];
            a1288 = (a1282 + (12*b150));
            s468 = X[a1288];
            s469 = X[(a1288 + 1)];
            a1289 = (a1282 + (14*b150));
            s470 = X[a1289];
            s471 = X[(a1289 + 1)];
            a1290 = (a1282 + (16*b150));
            s472 = X[a1290];
            s473 = X[(a1290 + 1)];
            a1291 = (a1282 + (18*b150));
            s474 = X[a1291];
            s475 = X[(a1291 + 1)];
            t1310 = (s460 + s472);
            t1311 = (s461 + s473);
            t1312 = (s460 - s472);
            t1313 = (s461 - s473);
            t1314 = (s464 + s468);
            t1315 = (s465 + s469);
            t1316 = (s464 - s468);
            t1317 = (s465 - s469);
            t1318 = (t1310 + t1314);
            t1319 = (t1311 + t1315);
            t1320 = (t1312 + t1317);
            t1321 = (t1313 - t1316);
            t1322 = (t1312 - t1317);
            t1323 = (t1313 + t1316);
            t1324 = (s456 + t1318);
            t1325 = (s457 + t1319);
            t1326 = (s456 - (0.25*t1318));
            t1327 = (s457 - (0.25*t1319));
            s476 = ((0.29389262614623657*t1320) + (0.47552825814757677*t1321));
            s477 = ((0.29389262614623657*t1321) - (0.47552825814757677*t1320));
            s478 = (0.55901699437494745*(t1310 - t1314));
            s479 = (0.55901699437494745*(t1311 - t1315));
            s480 = ((0.47552825814757682*t1323) - (0.29389262614623657*t1322));
            s481 = ((0.47552825814757682*t1322) + (0.29389262614623657*t1323));
            t1328 = (t1326 + s478);
            t1329 = (t1327 + s479);
            t1330 = (t1326 - s478);
            t1331 = (t1327 - s479);
            t1332 = (s476 + s480);
            t1333 = (s477 - s481);
            t1334 = (s476 - s480);
            t1335 = (s477 + s481);
            t1336 = (t1328 + t1332);
            t1337 = (t1329 + t1333);
            t1338 = (t1328 - t1332);
            t1339 = (t1329 - t1333);
            t1340 = (t1330 + t1335);
            t1341 = (t1331 - t1334);
            t1342 = (t1330 - t1335);
            t1343 = (t1331 + t1334);
            t1344 = (s462 + s474);
            t1345 = (s463 + s475);
            t1346 = (s462 - s474);
            t1347 = (s463 - s475);
            t1348 = (s466 + s470);
            t1349 = (s467 + s471);
            t1350 = (s466 - s470);
            t1351 = (s467 - s471);
            t1352 = (t1344 + t1348);
            t1353 = (t1345 + t1349);
            t1354 = (t1346 + t1351);
            t1355 = (t1347 - t1350);
            t1356 = (t1346 - t1351);
            t1357 = (t1347 + t1350);
            t1358 = (s458 + t1352);
            t1359 = (s459 + t1353);
            t1360 = (s458 - (0.25*t1352));
            t1361 = (s459 - (0.25*t1353));
            s482 = ((0.29389262614623657*t1354) + (0.47552825814757677*t1355));
            s483 = ((0.29389262614623657*t1355) - (0.47552825814757677*t1354));
            s484 = (0.55901699437494745*(t1344 - t1348));
            s485 = (0.55901699437494745*(t1345 - t1349));
            s486 = ((0.47552825814757682*t1357) - (0.29389262614623657*t1356));
            s487 = ((0.47552825814757682*t1356) + (0.29389262614623657*t1357));
            t1362 = (t1360 + s484);
            t1363 = (t1361 + s485);
            t1364 = (t1360 - s484);
            t1365 = (t1361 - s485);
            t1366 = (s482 + s486);
            t1367 = (s483 - s487);
            t1368 = (s482 - s486);
            t1369 = (s483 + s487);
            t1370 = (t1362 + t1366);
            t1371 = (t1363 + t1367);
            t1372 = (t1362 - t1366);
            t1373 = (t1363 - t1367);
            s488 = ((0.80901699437494745*t1370) + (0.58778525229247314*t1371));
            s489 = ((0.80901699437494745*t1371) - (0.58778525229247314*t1370));
            s490 = ((0.58778525229247314*t1373) - (0.80901699437494745*t1372));
            s491 = ((0.58778525229247314*t1372) + (0.80901699437494745*t1373));
            t1374 = (t1364 + t1369);
            t1375 = (t1365 - t1368);
            t1376 = (t1364 - t1369);
            t1377 = (t1365 + t1368);
            s492 = ((0.3090169943749474*t1374) + (0.95105651629515353*t1375));
            s493 = ((0.3090169943749474*t1375) - (0.95105651629515353*t1374));
            s494 = ((0.95105651629515353*t1377) - (0.3090169943749474*t1376));
            s495 = ((0.95105651629515353*t1376) + (0.3090169943749474*t1377));
            s496 = (t1324 - t1358);
            s497 = (t1325 - t1359);
            s498 = (t1336 + s488);
            s499 = (t1337 + s489);
            s500 = (t1336 - s488);
            s501 = (t1337 - s489);
            s502 = (t1340 + s492);
            s503 = (t1341 + s493);
            s504 = (t1340 - s492);
            s505 = (t1341 - s493);
            s506 = (t1342 + s494);
            s507 = (t1343 - s495);
            s508 = (t1342 - s494);
            s509 = (t1343 + s495);
            s510 = (t1338 + s490);
            s511 = (t1339 - s491);
            s512 = (t1338 - s490);
            s513 = (t1339 + s491);
            a1292 = (18*j1);
            a1293 = TW1[a1292];
            a1294 = TW1[(a1292 + 1)];
            a1295 = TW1[(a1292 + 2)];
            a1296 = TW1[(a1292 + 3)];
            a1297 = TW1[(a1292 + 4)];
            a1298 = TW1[(a1292 + 5)];
            a1299 = TW1[(a1292 + 6)];
            a1300 = TW1[(a1292 + 7)];
            a1301 = TW1[(a1292 + 8)];
            a1302 = TW1[(a1292 + 9)];
            a1303 = TW1[(a1292 + 10)];
            a1304 = TW1[(a1292 + 11)];
            a1305 = TW1[(a1292 + 12)];
            a1306 = TW1[(a1292 + 13)];
            a1307 = TW1[(a1292 + 14)];
            a1308 = TW1[(a1292 + 15)];
            a1309 = TW1[(a1292 + 16)];
            a1310 = TW1[(a1292 + 17)];
            a1311 = (20*a1281);
            a1312 = (a1280 + a1311);
            Y[a1312] = (t1324 + t1358);
            Y[(a1312 + 1)] = (t1325 + t1359);
            a1313 = (a1280 + (2*m1) + a1311);
            Y[a1313] = ((a1293*s498) - (a1294*s499));
            Y[(a1313 + 1)] = ((a1294*s498) + (a1293*s499));
            a1314 = (a1280 + (4*m1) + a1311);
            Y[a1314] = ((a1295*s502) - (a1296*s503));
            Y[(a1314 + 1)] = ((a1296*s502) + (a1295*s503));
            a1315 = (a1280 + (6*m1) + a1311);
            Y[a1315] = ((a1297*s506) - (a1298*s507));
            Y[(a1315 + 1)] = ((a1298*s506) + (a1297*s507));
            a1316 = (a1280 + (8*m1) + a1311);
            Y[a1316] = ((a1299*s510) - (a1300*s511));
            Y[(a1316 + 1)] = ((a1300*s510) + (a1299*s511));
            a1317 = (a1280 + (10*m1) + a1311);
            Y[a1317] = ((a1301*s496) - (a1302*s497));
            Y[(a1317 + 1)] = ((a1302*s496) + (a1301*s497));
            a1318 = (a1280 + (12*m1) + a1311);
            Y[a1318] = ((a1303*s500) - (a1304*s501));
            Y[(a1318 + 1)] = ((a1304*s500) + (a1303*s501));
            a1319 = (a1280 + (14*m1) + a1311);
            Y[a1319] = ((a1305*s504) - (a1306*s505));
            Y[(a1319 + 1)] = ((a1306*s504) + (a1305*s505));
            a1320 = (a1280 + (16*m1) + a1311);
            Y[a1320] = ((a1307*s508) - (a1308*s509));
            Y[(a1320 + 1)] = ((a1308*s508) + (a1307*s509));
            a1321 = (a1280 + (18*m1) + a1311);
            Y[a1321] = ((a1309*s512) - (a1310*s513));
            Y[(a1321 + 1)] = ((a1310*s512) + (a1309*s513));
        }
    }
}