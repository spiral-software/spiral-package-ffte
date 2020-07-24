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


void dft10c_(double  *Y, double  *X, int  *lp1, int  *mp1) {
    double s402, s403, s404, s405, s406, s407, s408, s409, 
            s410, s411, s412, s413, s414, s415, s416, s417, 
            s418, s419, s420, s421, s422, s423, s424, s425, 
            s426, s427, s428, s429, s430, s431, s432, s433, 
            s434, s435, s436, s437, s438, s439, s440, s441, 
            t1310, t1311, t1312, t1313, t1314, t1315, t1316, t1317, 
            t1318, t1319, t1320, t1321, t1322, t1323, t1324, t1325, 
            t1326, t1327, t1328, t1329, t1330, t1331, t1332, t1333, 
            t1334, t1335, t1336, t1337, t1338, t1339, t1340, t1341, 
            t1342, t1343, t1344, t1345, t1346, t1347, t1348, t1349, 
            t1350, t1351, t1352, t1353, t1354, t1355, t1356, t1357, 
            t1358, t1359, t1360, t1361, t1362, t1363, t1364, t1365, 
            t1366, t1367, t1368, t1369, t1370, t1371, t1372, t1373, 
            t1374, t1375, t1376, t1377;
    int a763, a764, a765, a766, a767, a768, a769, a770, 
            a771, a772, a773, a774, a775, a776, a777, a778, 
            a779, a780, a781, a782, b43, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int k1 = 0; k1 < m1; k1++) {
        a763 = (2*k1);
        s402 = X[a763];
        a764 = (a763 + 1);
        s403 = X[a764];
        b43 = (l1*m1);
        a765 = (a763 + (2*b43));
        s404 = X[a765];
        s405 = X[(a765 + 1)];
        a766 = (a763 + (4*b43));
        s406 = X[a766];
        s407 = X[(a766 + 1)];
        a767 = (a763 + (6*b43));
        s408 = X[a767];
        s409 = X[(a767 + 1)];
        a768 = (a763 + (8*b43));
        s410 = X[a768];
        s411 = X[(a768 + 1)];
        a769 = (a763 + (10*b43));
        s412 = X[a769];
        s413 = X[(a769 + 1)];
        a770 = (a763 + (12*b43));
        s414 = X[a770];
        s415 = X[(a770 + 1)];
        a771 = (a763 + (14*b43));
        s416 = X[a771];
        s417 = X[(a771 + 1)];
        a772 = (a763 + (16*b43));
        s418 = X[a772];
        s419 = X[(a772 + 1)];
        a773 = (a763 + (18*b43));
        s420 = X[a773];
        s421 = X[(a773 + 1)];
        t1310 = (s406 + s418);
        t1311 = (s407 + s419);
        t1312 = (s406 - s418);
        t1313 = (s407 - s419);
        t1314 = (s410 + s414);
        t1315 = (s411 + s415);
        t1316 = (s410 - s414);
        t1317 = (s411 - s415);
        t1318 = (t1310 + t1314);
        t1319 = (t1311 + t1315);
        t1320 = (t1312 + t1317);
        t1321 = (t1313 - t1316);
        t1322 = (t1312 - t1317);
        t1323 = (t1313 + t1316);
        t1324 = (s402 + t1318);
        t1325 = (s403 + t1319);
        t1326 = (s402 - (0.25*t1318));
        t1327 = (s403 - (0.25*t1319));
        s422 = ((0.29389262614623657*t1320) + (0.47552825814757677*t1321));
        s423 = ((0.29389262614623657*t1321) - (0.47552825814757677*t1320));
        s424 = (0.55901699437494745*(t1310 - t1314));
        s425 = (0.55901699437494745*(t1311 - t1315));
        s426 = ((0.47552825814757682*t1323) - (0.29389262614623657*t1322));
        s427 = ((0.47552825814757682*t1322) + (0.29389262614623657*t1323));
        t1328 = (t1326 + s424);
        t1329 = (t1327 + s425);
        t1330 = (t1326 - s424);
        t1331 = (t1327 - s425);
        t1332 = (s422 + s426);
        t1333 = (s423 - s427);
        t1334 = (s422 - s426);
        t1335 = (s423 + s427);
        t1336 = (t1328 + t1332);
        t1337 = (t1329 + t1333);
        t1338 = (t1328 - t1332);
        t1339 = (t1329 - t1333);
        t1340 = (t1330 + t1335);
        t1341 = (t1331 - t1334);
        t1342 = (t1330 - t1335);
        t1343 = (t1331 + t1334);
        t1344 = (s408 + s420);
        t1345 = (s409 + s421);
        t1346 = (s408 - s420);
        t1347 = (s409 - s421);
        t1348 = (s412 + s416);
        t1349 = (s413 + s417);
        t1350 = (s412 - s416);
        t1351 = (s413 - s417);
        t1352 = (t1344 + t1348);
        t1353 = (t1345 + t1349);
        t1354 = (t1346 + t1351);
        t1355 = (t1347 - t1350);
        t1356 = (t1346 - t1351);
        t1357 = (t1347 + t1350);
        t1358 = (s404 + t1352);
        t1359 = (s405 + t1353);
        t1360 = (s404 - (0.25*t1352));
        t1361 = (s405 - (0.25*t1353));
        s428 = ((0.29389262614623657*t1354) + (0.47552825814757677*t1355));
        s429 = ((0.29389262614623657*t1355) - (0.47552825814757677*t1354));
        s430 = (0.55901699437494745*(t1344 - t1348));
        s431 = (0.55901699437494745*(t1345 - t1349));
        s432 = ((0.47552825814757682*t1357) - (0.29389262614623657*t1356));
        s433 = ((0.47552825814757682*t1356) + (0.29389262614623657*t1357));
        t1362 = (t1360 + s430);
        t1363 = (t1361 + s431);
        t1364 = (t1360 - s430);
        t1365 = (t1361 - s431);
        t1366 = (s428 + s432);
        t1367 = (s429 - s433);
        t1368 = (s428 - s432);
        t1369 = (s429 + s433);
        t1370 = (t1362 + t1366);
        t1371 = (t1363 + t1367);
        t1372 = (t1362 - t1366);
        t1373 = (t1363 - t1367);
        s434 = ((0.80901699437494745*t1370) + (0.58778525229247314*t1371));
        s435 = ((0.80901699437494745*t1371) - (0.58778525229247314*t1370));
        s436 = ((0.58778525229247314*t1373) - (0.80901699437494745*t1372));
        s437 = ((0.58778525229247314*t1372) + (0.80901699437494745*t1373));
        t1374 = (t1364 + t1369);
        t1375 = (t1365 - t1368);
        t1376 = (t1364 - t1369);
        t1377 = (t1365 + t1368);
        s438 = ((0.3090169943749474*t1374) + (0.95105651629515353*t1375));
        s439 = ((0.3090169943749474*t1375) - (0.95105651629515353*t1374));
        s440 = ((0.95105651629515353*t1377) - (0.3090169943749474*t1376));
        s441 = ((0.95105651629515353*t1376) + (0.3090169943749474*t1377));
        Y[a763] = (t1324 + t1358);
        Y[a764] = (t1325 + t1359);
        a774 = (a763 + (2*m1));
        Y[a774] = (t1336 + s434);
        Y[(a774 + 1)] = (t1337 + s435);
        a775 = (a763 + (4*m1));
        Y[a775] = (t1340 + s438);
        Y[(a775 + 1)] = (t1341 + s439);
        a776 = (a763 + (6*m1));
        Y[a776] = (t1342 + s440);
        Y[(a776 + 1)] = (t1343 - s441);
        a777 = (a763 + (8*m1));
        Y[a777] = (t1338 + s436);
        Y[(a777 + 1)] = (t1339 - s437);
        a778 = (a763 + (10*m1));
        Y[a778] = (t1324 - t1358);
        Y[(a778 + 1)] = (t1325 - t1359);
        a779 = (a763 + (12*m1));
        Y[a779] = (t1336 - s434);
        Y[(a779 + 1)] = (t1337 - s435);
        a780 = (a763 + (14*m1));
        Y[a780] = (t1340 - s438);
        Y[(a780 + 1)] = (t1341 - s439);
        a781 = (a763 + (16*m1));
        Y[a781] = (t1342 - s440);
        Y[(a781 + 1)] = (t1343 + s441);
        a782 = (a763 + (18*m1));
        Y[a782] = (t1338 - s436);
        Y[(a782 + 1)] = (t1339 + s437);
    }
}