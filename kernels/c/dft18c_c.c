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


void dft18c_(double  *Y, double  *X, int  *lp1, int  *mp1) {
    static double D39[12];
    static double D40[24];
    double s1422, s1423, s1424, s1425, s1426, s1427, s1428, s1429, 
            s1430, s1431, s1432, s1433, s1434, s1435, s1436, s1437, 
            s1438, s1439, s1440, s1441, s1442, s1443, s1444, s1445, 
            s1446, s1447, s1448, s1449, s1450, s1451, s1452, s1453, 
            s1454, s1455, s1456, s1457, s1458, s1459, s1460, s1461, 
            s1462, s1463, s1464, s1465, s1466, s1467, s1468, s1469, 
            s1470, s1471, s1472, s1473, s1474, s1475, s1476, s1477, 
            s1478, s1479, s1480, s1481, s1482, s1483, s1484, s1485, 
            s1486, s1487, s1488, s1489, s1490, s1491, s1492, s1493, 
            s1494, s1495, s1496, s1497, s1498, s1499, s1500, s1501, 
            s1502, s1503, s1504, s1505, s1506, s1507, s1508, s1509, 
            s1510, s1511, s1512, s1513, s1514, s1515, s1516, s1517, 
            s1518, s1519, s1520, s1521, s1522, s1523, s1524, s1525, 
            s1526, s1527, s1528, s1529, t2074, t2075, t2076, t2077, 
            t2078, t2079, t2080, t2081, t2082, t2083, t2084, t2085, 
            t2086, t2087, t2088, t2089, t2090, t2091, t2092, t2093, 
            t2094, t2095, t2096, t2097, t2098, t2099, t2100, t2101, 
            t2102, t2103, t2104, t2105, t2106, t2107, t2108, t2109, 
            t2110, t2111, t2112, t2113, t2114, t2115, t2116, t2117, 
            t2118, t2119, t2120, t2121, t2122, t2123, t2124, t2125, 
            t2126, t2127, t2128, t2129, t2130, t2131, t2132, t2133, 
            t2134, t2135, t2136, t2137, t2138, t2139, t2140, t2141, 
            t2142, t2143, t2144, t2145, t2146, t2147, t2148, t2149, 
            t2150, t2151, t2152, t2153, t2154, t2155, t2156, t2157, 
            t2158, t2159, t2160, t2161, t2162, t2163, t2164, t2165, 
            t2166, t2167, t2168, t2169, t2170, t2171, t2172, t2173, 
            t2174, t2175, t2176, t2177, t2178, t2179, t2180, t2181, 
            t2182, t2183, t2184, t2185, t2186, t2187, t2188, t2189, 
            t2190, t2191, t2192, t2193;
    int a3019, a3020, a3021, a3022, a3023, a3024, a3025, a3026, 
            a3027, a3028, a3029, a3030, a3031, a3032, a3033, a3034, 
            a3035, a3036, a3037, a3038, a3039, a3040, a3041, a3042, 
            a3043, a3044, a3045, a3046, a3047, a3048, a3049, a3050, 
            a3051, a3052, a3053, a3054, b75, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int k1 = 0; k1 < m1; k1++) {
        a3019 = (2*k1);
        s1422 = X[a3019];
        a3020 = (a3019 + 1);
        s1423 = X[a3020];
        b75 = (l1*m1);
        a3021 = (a3019 + (2*b75));
        s1424 = X[a3021];
        s1425 = X[(a3021 + 1)];
        a3022 = (a3019 + (4*b75));
        s1426 = X[a3022];
        s1427 = X[(a3022 + 1)];
        a3023 = (a3019 + (6*b75));
        s1428 = X[a3023];
        s1429 = X[(a3023 + 1)];
        a3024 = (a3019 + (8*b75));
        s1430 = X[a3024];
        s1431 = X[(a3024 + 1)];
        a3025 = (a3019 + (10*b75));
        s1432 = X[a3025];
        s1433 = X[(a3025 + 1)];
        a3026 = (a3019 + (12*b75));
        s1434 = X[a3026];
        s1435 = X[(a3026 + 1)];
        a3027 = (a3019 + (14*b75));
        s1436 = X[a3027];
        s1437 = X[(a3027 + 1)];
        a3028 = (a3019 + (16*b75));
        s1438 = X[a3028];
        s1439 = X[(a3028 + 1)];
        a3029 = (a3019 + (18*b75));
        s1440 = X[a3029];
        s1441 = X[(a3029 + 1)];
        a3030 = (a3019 + (20*b75));
        s1442 = X[a3030];
        s1443 = X[(a3030 + 1)];
        a3031 = (a3019 + (22*b75));
        s1444 = X[a3031];
        s1445 = X[(a3031 + 1)];
        a3032 = (a3019 + (24*b75));
        s1446 = X[a3032];
        s1447 = X[(a3032 + 1)];
        a3033 = (a3019 + (26*b75));
        s1448 = X[a3033];
        s1449 = X[(a3033 + 1)];
        a3034 = (a3019 + (28*b75));
        s1450 = X[a3034];
        s1451 = X[(a3034 + 1)];
        a3035 = (a3019 + (30*b75));
        s1452 = X[a3035];
        s1453 = X[(a3035 + 1)];
        a3036 = (a3019 + (32*b75));
        s1454 = X[a3036];
        s1455 = X[(a3036 + 1)];
        a3037 = (a3019 + (34*b75));
        s1456 = X[a3037];
        s1457 = X[(a3037 + 1)];
        t2074 = (s1434 + s1446);
        t2075 = (s1435 + s1447);
        t2076 = (s1422 + t2074);
        t2077 = (s1423 + t2075);
        t2078 = (s1422 - (0.5*t2074));
        t2079 = (s1423 - (0.5*t2075));
        s1458 = (0.8660254037844386*(s1435 - s1447));
        s1459 = (0.8660254037844386*(s1434 - s1446));
        s1460 = ((D39[0]*t2076) - (D39[1]*t2077));
        s1461 = ((D39[1]*t2076) + (D39[0]*t2077));
        t2080 = (t2078 + s1458);
        t2081 = (t2079 - s1459);
        t2082 = (t2078 - s1458);
        t2083 = (t2079 + s1459);
        s1462 = ((D40[0]*t2080) - (D40[1]*t2081));
        s1463 = ((D40[1]*t2080) + (D40[0]*t2081));
        s1464 = ((D40[2]*t2082) - (D40[3]*t2083));
        s1465 = ((D40[3]*t2082) + (D40[2]*t2083));
        t2084 = (s1436 + s1448);
        t2085 = (s1437 + s1449);
        t2086 = (s1424 + t2084);
        t2087 = (s1425 + t2085);
        t2088 = (s1424 - (0.5*t2084));
        t2089 = (s1425 - (0.5*t2085));
        s1466 = (0.8660254037844386*(s1437 - s1449));
        s1467 = (0.8660254037844386*(s1436 - s1448));
        s1468 = ((D39[2]*t2086) - (D39[3]*t2087));
        s1469 = ((D39[3]*t2086) + (D39[2]*t2087));
        t2090 = (t2088 + s1466);
        t2091 = (t2089 - s1467);
        t2092 = (t2088 - s1466);
        t2093 = (t2089 + s1467);
        s1470 = ((D40[4]*t2090) - (D40[5]*t2091));
        s1471 = ((D40[5]*t2090) + (D40[4]*t2091));
        s1472 = ((D40[6]*t2092) - (D40[7]*t2093));
        s1473 = ((D40[7]*t2092) + (D40[6]*t2093));
        t2094 = (s1438 + s1450);
        t2095 = (s1439 + s1451);
        t2096 = (s1426 + t2094);
        t2097 = (s1427 + t2095);
        t2098 = (s1426 - (0.5*t2094));
        t2099 = (s1427 - (0.5*t2095));
        s1474 = (0.8660254037844386*(s1439 - s1451));
        s1475 = (0.8660254037844386*(s1438 - s1450));
        s1476 = ((D39[4]*t2096) - (D39[5]*t2097));
        s1477 = ((D39[5]*t2096) + (D39[4]*t2097));
        t2100 = (t2098 + s1474);
        t2101 = (t2099 - s1475);
        t2102 = (t2098 - s1474);
        t2103 = (t2099 + s1475);
        s1478 = ((D40[8]*t2100) - (D40[9]*t2101));
        s1479 = ((D40[9]*t2100) + (D40[8]*t2101));
        s1480 = ((D40[10]*t2102) - (D40[11]*t2103));
        s1481 = ((D40[11]*t2102) + (D40[10]*t2103));
        t2104 = (s1440 + s1452);
        t2105 = (s1441 + s1453);
        t2106 = (s1428 + t2104);
        t2107 = (s1429 + t2105);
        t2108 = (s1428 - (0.5*t2104));
        t2109 = (s1429 - (0.5*t2105));
        s1482 = (0.8660254037844386*(s1441 - s1453));
        s1483 = (0.8660254037844386*(s1440 - s1452));
        s1484 = ((D39[6]*t2106) - (D39[7]*t2107));
        s1485 = ((D39[7]*t2106) + (D39[6]*t2107));
        t2110 = (t2108 + s1482);
        t2111 = (t2109 - s1483);
        t2112 = (t2108 - s1482);
        t2113 = (t2109 + s1483);
        s1486 = ((D40[12]*t2110) - (D40[13]*t2111));
        s1487 = ((D40[13]*t2110) + (D40[12]*t2111));
        s1488 = ((D40[14]*t2112) - (D40[15]*t2113));
        s1489 = ((D40[15]*t2112) + (D40[14]*t2113));
        t2114 = (s1442 + s1454);
        t2115 = (s1443 + s1455);
        t2116 = (s1430 + t2114);
        t2117 = (s1431 + t2115);
        t2118 = (s1430 - (0.5*t2114));
        t2119 = (s1431 - (0.5*t2115));
        s1490 = (0.8660254037844386*(s1443 - s1455));
        s1491 = (0.8660254037844386*(s1442 - s1454));
        s1492 = ((D39[8]*t2116) - (D39[9]*t2117));
        s1493 = ((D39[9]*t2116) + (D39[8]*t2117));
        t2120 = (t2118 + s1490);
        t2121 = (t2119 - s1491);
        t2122 = (t2118 - s1490);
        t2123 = (t2119 + s1491);
        s1494 = ((D40[16]*t2120) - (D40[17]*t2121));
        s1495 = ((D40[17]*t2120) + (D40[16]*t2121));
        s1496 = ((D40[18]*t2122) - (D40[19]*t2123));
        s1497 = ((D40[19]*t2122) + (D40[18]*t2123));
        t2124 = (s1444 + s1456);
        t2125 = (s1445 + s1457);
        t2126 = (s1432 + t2124);
        t2127 = (s1433 + t2125);
        t2128 = (s1432 - (0.5*t2124));
        t2129 = (s1433 - (0.5*t2125));
        s1498 = (0.8660254037844386*(s1445 - s1457));
        s1499 = (0.8660254037844386*(s1444 - s1456));
        s1500 = ((D39[10]*t2126) - (D39[11]*t2127));
        s1501 = ((D39[11]*t2126) + (D39[10]*t2127));
        t2130 = (t2128 + s1498);
        t2131 = (t2129 - s1499);
        t2132 = (t2128 - s1498);
        t2133 = (t2129 + s1499);
        s1502 = ((D40[20]*t2130) - (D40[21]*t2131));
        s1503 = ((D40[21]*t2130) + (D40[20]*t2131));
        s1504 = ((D40[22]*t2132) - (D40[23]*t2133));
        s1505 = ((D40[23]*t2132) + (D40[22]*t2133));
        t2134 = (s1476 + s1492);
        t2135 = (s1477 + s1493);
        t2136 = (s1460 + t2134);
        t2137 = (s1461 + t2135);
        t2138 = (s1460 - (0.5*t2134));
        t2139 = (s1461 - (0.5*t2135));
        s1506 = (0.8660254037844386*(s1477 - s1493));
        s1507 = (0.8660254037844386*(s1476 - s1492));
        t2140 = (t2138 + s1506);
        t2141 = (t2139 - s1507);
        t2142 = (t2138 - s1506);
        t2143 = (t2139 + s1507);
        t2144 = (s1484 + s1500);
        t2145 = (s1485 + s1501);
        t2146 = (s1468 + t2144);
        t2147 = (s1469 + t2145);
        t2148 = (s1468 - (0.5*t2144));
        t2149 = (s1469 - (0.5*t2145));
        s1508 = (0.8660254037844386*(s1485 - s1501));
        s1509 = (0.8660254037844386*(s1484 - s1500));
        t2150 = (t2148 + s1508);
        t2151 = (t2149 - s1509);
        t2152 = (t2148 - s1508);
        t2153 = (t2149 + s1509);
        s1510 = ((0.5*t2150) + (0.8660254037844386*t2151));
        s1511 = ((0.5*t2151) - (0.8660254037844386*t2150));
        s1512 = ((0.8660254037844386*t2153) - (0.5*t2152));
        s1513 = ((0.8660254037844386*t2152) + (0.5*t2153));
        t2154 = (s1478 + s1494);
        t2155 = (s1479 + s1495);
        t2156 = (s1462 + t2154);
        t2157 = (s1463 + t2155);
        t2158 = (s1462 - (0.5*t2154));
        t2159 = (s1463 - (0.5*t2155));
        s1514 = (0.8660254037844386*(s1479 - s1495));
        s1515 = (0.8660254037844386*(s1478 - s1494));
        t2160 = (t2158 + s1514);
        t2161 = (t2159 - s1515);
        t2162 = (t2158 - s1514);
        t2163 = (t2159 + s1515);
        t2164 = (s1486 + s1502);
        t2165 = (s1487 + s1503);
        t2166 = (s1470 + t2164);
        t2167 = (s1471 + t2165);
        t2168 = (s1470 - (0.5*t2164));
        t2169 = (s1471 - (0.5*t2165));
        s1516 = (0.8660254037844386*(s1487 - s1503));
        s1517 = (0.8660254037844386*(s1486 - s1502));
        t2170 = (t2168 + s1516);
        t2171 = (t2169 - s1517);
        t2172 = (t2168 - s1516);
        t2173 = (t2169 + s1517);
        s1518 = ((0.5*t2170) + (0.8660254037844386*t2171));
        s1519 = ((0.5*t2171) - (0.8660254037844386*t2170));
        s1520 = ((0.8660254037844386*t2173) - (0.5*t2172));
        s1521 = ((0.8660254037844386*t2172) + (0.5*t2173));
        t2174 = (s1480 + s1496);
        t2175 = (s1481 + s1497);
        t2176 = (s1464 + t2174);
        t2177 = (s1465 + t2175);
        t2178 = (s1464 - (0.5*t2174));
        t2179 = (s1465 - (0.5*t2175));
        s1522 = (0.8660254037844386*(s1481 - s1497));
        s1523 = (0.8660254037844386*(s1480 - s1496));
        t2180 = (t2178 + s1522);
        t2181 = (t2179 - s1523);
        t2182 = (t2178 - s1522);
        t2183 = (t2179 + s1523);
        t2184 = (s1488 + s1504);
        t2185 = (s1489 + s1505);
        t2186 = (s1472 + t2184);
        t2187 = (s1473 + t2185);
        t2188 = (s1472 - (0.5*t2184));
        t2189 = (s1473 - (0.5*t2185));
        s1524 = (0.8660254037844386*(s1489 - s1505));
        s1525 = (0.8660254037844386*(s1488 - s1504));
        t2190 = (t2188 + s1524);
        t2191 = (t2189 - s1525);
        t2192 = (t2188 - s1524);
        t2193 = (t2189 + s1525);
        s1526 = ((0.5*t2190) + (0.8660254037844386*t2191));
        s1527 = ((0.5*t2191) - (0.8660254037844386*t2190));
        s1528 = ((0.8660254037844386*t2193) - (0.5*t2192));
        s1529 = ((0.8660254037844386*t2192) + (0.5*t2193));
        Y[a3019] = (t2136 + t2146);
        Y[a3020] = (t2137 + t2147);
        a3038 = (a3019 + (2*m1));
        Y[a3038] = (t2156 + t2166);
        Y[(a3038 + 1)] = (t2157 + t2167);
        a3039 = (a3019 + (4*m1));
        Y[a3039] = (t2176 + t2186);
        Y[(a3039 + 1)] = (t2177 + t2187);
        a3040 = (a3019 + (6*m1));
        Y[a3040] = (t2140 + s1510);
        Y[(a3040 + 1)] = (t2141 + s1511);
        a3041 = (a3019 + (8*m1));
        Y[a3041] = (t2160 + s1518);
        Y[(a3041 + 1)] = (t2161 + s1519);
        a3042 = (a3019 + (10*m1));
        Y[a3042] = (t2180 + s1526);
        Y[(a3042 + 1)] = (t2181 + s1527);
        a3043 = (a3019 + (12*m1));
        Y[a3043] = (t2142 + s1512);
        Y[(a3043 + 1)] = (t2143 - s1513);
        a3044 = (a3019 + (14*m1));
        Y[a3044] = (t2162 + s1520);
        Y[(a3044 + 1)] = (t2163 - s1521);
        a3045 = (a3019 + (16*m1));
        Y[a3045] = (t2182 + s1528);
        Y[(a3045 + 1)] = (t2183 - s1529);
        a3046 = (a3019 + (18*m1));
        Y[a3046] = (t2136 - t2146);
        Y[(a3046 + 1)] = (t2137 - t2147);
        a3047 = (a3019 + (20*m1));
        Y[a3047] = (t2156 - t2166);
        Y[(a3047 + 1)] = (t2157 - t2167);
        a3048 = (a3019 + (22*m1));
        Y[a3048] = (t2176 - t2186);
        Y[(a3048 + 1)] = (t2177 - t2187);
        a3049 = (a3019 + (24*m1));
        Y[a3049] = (t2140 - s1510);
        Y[(a3049 + 1)] = (t2141 - s1511);
        a3050 = (a3019 + (26*m1));
        Y[a3050] = (t2160 - s1518);
        Y[(a3050 + 1)] = (t2161 - s1519);
        a3051 = (a3019 + (28*m1));
        Y[a3051] = (t2180 - s1526);
        Y[(a3051 + 1)] = (t2181 - s1527);
        a3052 = (a3019 + (30*m1));
        Y[a3052] = (t2142 - s1512);
        Y[(a3052 + 1)] = (t2143 + s1513);
        a3053 = (a3019 + (32*m1));
        Y[a3053] = (t2162 - s1520);
        Y[(a3053 + 1)] = (t2163 + s1521);
        a3054 = (a3019 + (34*m1));
        Y[a3054] = (t2182 - s1528);
        Y[(a3054 + 1)] = (t2183 + s1529);
    }
}