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


void dft18b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    static double D39[12];
    static double D40[24];
    double a3989, a3990, a3991, a3992, a3993, a3994, a3995, a3996, 
            a3997, a3998, a3999, a4000, a4001, a4002, a4003, a4004, 
            a4005, a4006, a4007, a4008, a4009, a4010, a4011, a4012, 
            a4013, a4014, a4015, a4016, a4017, a4018, a4019, a4020, 
            a4021, a4022, s1524, s1525, s1526, s1527, s1528, s1529, 
            s1530, s1531, s1532, s1533, s1534, s1535, s1536, s1537, 
            s1538, s1539, s1540, s1541, s1542, s1543, s1544, s1545, 
            s1546, s1547, s1548, s1549, s1550, s1551, s1552, s1553, 
            s1554, s1555, s1556, s1557, s1558, s1559, s1560, s1561, 
            s1562, s1563, s1564, s1565, s1566, s1567, s1568, s1569, 
            s1570, s1571, s1572, s1573, s1574, s1575, s1576, s1577, 
            s1578, s1579, s1580, s1581, s1582, s1583, s1584, s1585, 
            s1586, s1587, s1588, s1589, s1590, s1591, s1592, s1593, 
            s1594, s1595, s1596, s1597, s1598, s1599, s1600, s1601, 
            s1602, s1603, s1604, s1605, s1606, s1607, s1608, s1609, 
            s1610, s1611, s1612, s1613, s1614, s1615, s1616, s1617, 
            s1618, s1619, s1620, s1621, s1622, s1623, s1624, s1625, 
            s1626, s1627, s1628, s1629, s1630, s1631, s1632, s1633, 
            s1634, s1635, s1636, s1637, s1638, s1639, s1640, s1641, 
            s1642, s1643, s1644, s1645, s1646, s1647, s1648, s1649, 
            s1650, s1651, s1652, s1653, s1654, s1655, s1656, s1657, 
            s1658, s1659, s1660, s1661, s1662, s1663, s1664, s1665, 
            t2074, t2075, t2076, t2077, t2078, t2079, t2080, t2081, 
            t2082, t2083, t2084, t2085, t2086, t2087, t2088, t2089, 
            t2090, t2091, t2092, t2093, t2094, t2095, t2096, t2097, 
            t2098, t2099, t2100, t2101, t2102, t2103, t2104, t2105, 
            t2106, t2107, t2108, t2109, t2110, t2111, t2112, t2113, 
            t2114, t2115, t2116, t2117, t2118, t2119, t2120, t2121, 
            t2122, t2123, t2124, t2125, t2126, t2127, t2128, t2129, 
            t2130, t2131, t2132, t2133, t2134, t2135, t2136, t2137, 
            t2138, t2139, t2140, t2141, t2142, t2143, t2144, t2145, 
            t2146, t2147, t2148, t2149, t2150, t2151, t2152, t2153, 
            t2154, t2155, t2156, t2157, t2158, t2159, t2160, t2161, 
            t2162, t2163, t2164, t2165, t2166, t2167, t2168, t2169, 
            t2170, t2171, t2172, t2173, t2174, t2175, t2176, t2177, 
            t2178, t2179, t2180, t2181, t2182, t2183, t2184, t2185, 
            t2186, t2187, t2188, t2189, t2190, t2191, t2192, t2193;
    int a3968, a3969, a3970, a3971, a3972, a3973, a3974, a3975, 
            a3976, a3977, a3978, a3979, a3980, a3981, a3982, a3983, 
            a3984, a3985, a3986, a3987, a3988, a4023, a4024, a4025, 
            a4026, a4027, a4028, a4029, a4030, a4031, a4032, a4033, 
            a4034, a4035, a4036, a4037, a4038, a4039, a4040, a4041, 
            b270, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a3968 = (2*k1);
            a3969 = (j1*m1);
            a3970 = (a3968 + (2*a3969));
            s1524 = X[a3970];
            s1525 = X[(a3970 + 1)];
            b270 = (l1*m1);
            a3971 = (a3970 + (2*b270));
            s1526 = X[a3971];
            s1527 = X[(a3971 + 1)];
            a3972 = (a3970 + (4*b270));
            s1528 = X[a3972];
            s1529 = X[(a3972 + 1)];
            a3973 = (a3970 + (6*b270));
            s1530 = X[a3973];
            s1531 = X[(a3973 + 1)];
            a3974 = (a3970 + (8*b270));
            s1532 = X[a3974];
            s1533 = X[(a3974 + 1)];
            a3975 = (a3970 + (10*b270));
            s1534 = X[a3975];
            s1535 = X[(a3975 + 1)];
            a3976 = (a3970 + (12*b270));
            s1536 = X[a3976];
            s1537 = X[(a3976 + 1)];
            a3977 = (a3970 + (14*b270));
            s1538 = X[a3977];
            s1539 = X[(a3977 + 1)];
            a3978 = (a3970 + (16*b270));
            s1540 = X[a3978];
            s1541 = X[(a3978 + 1)];
            a3979 = (a3970 + (18*b270));
            s1542 = X[a3979];
            s1543 = X[(a3979 + 1)];
            a3980 = (a3970 + (20*b270));
            s1544 = X[a3980];
            s1545 = X[(a3980 + 1)];
            a3981 = (a3970 + (22*b270));
            s1546 = X[a3981];
            s1547 = X[(a3981 + 1)];
            a3982 = (a3970 + (24*b270));
            s1548 = X[a3982];
            s1549 = X[(a3982 + 1)];
            a3983 = (a3970 + (26*b270));
            s1550 = X[a3983];
            s1551 = X[(a3983 + 1)];
            a3984 = (a3970 + (28*b270));
            s1552 = X[a3984];
            s1553 = X[(a3984 + 1)];
            a3985 = (a3970 + (30*b270));
            s1554 = X[a3985];
            s1555 = X[(a3985 + 1)];
            a3986 = (a3970 + (32*b270));
            s1556 = X[a3986];
            s1557 = X[(a3986 + 1)];
            a3987 = (a3970 + (34*b270));
            s1558 = X[a3987];
            s1559 = X[(a3987 + 1)];
            t2074 = (s1536 + s1548);
            t2075 = (s1537 + s1549);
            t2076 = (s1524 + t2074);
            t2077 = (s1525 + t2075);
            t2078 = (s1524 - (0.5*t2074));
            t2079 = (s1525 - (0.5*t2075));
            s1560 = (0.8660254037844386*(s1537 - s1549));
            s1561 = (0.8660254037844386*(s1536 - s1548));
            s1562 = ((D39[0]*t2076) - (D39[1]*t2077));
            s1563 = ((D39[1]*t2076) + (D39[0]*t2077));
            t2080 = (t2078 + s1560);
            t2081 = (t2079 - s1561);
            t2082 = (t2078 - s1560);
            t2083 = (t2079 + s1561);
            s1564 = ((D40[0]*t2080) - (D40[1]*t2081));
            s1565 = ((D40[1]*t2080) + (D40[0]*t2081));
            s1566 = ((D40[2]*t2082) - (D40[3]*t2083));
            s1567 = ((D40[3]*t2082) + (D40[2]*t2083));
            t2084 = (s1538 + s1550);
            t2085 = (s1539 + s1551);
            t2086 = (s1526 + t2084);
            t2087 = (s1527 + t2085);
            t2088 = (s1526 - (0.5*t2084));
            t2089 = (s1527 - (0.5*t2085));
            s1568 = (0.8660254037844386*(s1539 - s1551));
            s1569 = (0.8660254037844386*(s1538 - s1550));
            s1570 = ((D39[2]*t2086) - (D39[3]*t2087));
            s1571 = ((D39[3]*t2086) + (D39[2]*t2087));
            t2090 = (t2088 + s1568);
            t2091 = (t2089 - s1569);
            t2092 = (t2088 - s1568);
            t2093 = (t2089 + s1569);
            s1572 = ((D40[4]*t2090) - (D40[5]*t2091));
            s1573 = ((D40[5]*t2090) + (D40[4]*t2091));
            s1574 = ((D40[6]*t2092) - (D40[7]*t2093));
            s1575 = ((D40[7]*t2092) + (D40[6]*t2093));
            t2094 = (s1540 + s1552);
            t2095 = (s1541 + s1553);
            t2096 = (s1528 + t2094);
            t2097 = (s1529 + t2095);
            t2098 = (s1528 - (0.5*t2094));
            t2099 = (s1529 - (0.5*t2095));
            s1576 = (0.8660254037844386*(s1541 - s1553));
            s1577 = (0.8660254037844386*(s1540 - s1552));
            s1578 = ((D39[4]*t2096) - (D39[5]*t2097));
            s1579 = ((D39[5]*t2096) + (D39[4]*t2097));
            t2100 = (t2098 + s1576);
            t2101 = (t2099 - s1577);
            t2102 = (t2098 - s1576);
            t2103 = (t2099 + s1577);
            s1580 = ((D40[8]*t2100) - (D40[9]*t2101));
            s1581 = ((D40[9]*t2100) + (D40[8]*t2101));
            s1582 = ((D40[10]*t2102) - (D40[11]*t2103));
            s1583 = ((D40[11]*t2102) + (D40[10]*t2103));
            t2104 = (s1542 + s1554);
            t2105 = (s1543 + s1555);
            t2106 = (s1530 + t2104);
            t2107 = (s1531 + t2105);
            t2108 = (s1530 - (0.5*t2104));
            t2109 = (s1531 - (0.5*t2105));
            s1584 = (0.8660254037844386*(s1543 - s1555));
            s1585 = (0.8660254037844386*(s1542 - s1554));
            s1586 = ((D39[6]*t2106) - (D39[7]*t2107));
            s1587 = ((D39[7]*t2106) + (D39[6]*t2107));
            t2110 = (t2108 + s1584);
            t2111 = (t2109 - s1585);
            t2112 = (t2108 - s1584);
            t2113 = (t2109 + s1585);
            s1588 = ((D40[12]*t2110) - (D40[13]*t2111));
            s1589 = ((D40[13]*t2110) + (D40[12]*t2111));
            s1590 = ((D40[14]*t2112) - (D40[15]*t2113));
            s1591 = ((D40[15]*t2112) + (D40[14]*t2113));
            t2114 = (s1544 + s1556);
            t2115 = (s1545 + s1557);
            t2116 = (s1532 + t2114);
            t2117 = (s1533 + t2115);
            t2118 = (s1532 - (0.5*t2114));
            t2119 = (s1533 - (0.5*t2115));
            s1592 = (0.8660254037844386*(s1545 - s1557));
            s1593 = (0.8660254037844386*(s1544 - s1556));
            s1594 = ((D39[8]*t2116) - (D39[9]*t2117));
            s1595 = ((D39[9]*t2116) + (D39[8]*t2117));
            t2120 = (t2118 + s1592);
            t2121 = (t2119 - s1593);
            t2122 = (t2118 - s1592);
            t2123 = (t2119 + s1593);
            s1596 = ((D40[16]*t2120) - (D40[17]*t2121));
            s1597 = ((D40[17]*t2120) + (D40[16]*t2121));
            s1598 = ((D40[18]*t2122) - (D40[19]*t2123));
            s1599 = ((D40[19]*t2122) + (D40[18]*t2123));
            t2124 = (s1546 + s1558);
            t2125 = (s1547 + s1559);
            t2126 = (s1534 + t2124);
            t2127 = (s1535 + t2125);
            t2128 = (s1534 - (0.5*t2124));
            t2129 = (s1535 - (0.5*t2125));
            s1600 = (0.8660254037844386*(s1547 - s1559));
            s1601 = (0.8660254037844386*(s1546 - s1558));
            s1602 = ((D39[10]*t2126) - (D39[11]*t2127));
            s1603 = ((D39[11]*t2126) + (D39[10]*t2127));
            t2130 = (t2128 + s1600);
            t2131 = (t2129 - s1601);
            t2132 = (t2128 - s1600);
            t2133 = (t2129 + s1601);
            s1604 = ((D40[20]*t2130) - (D40[21]*t2131));
            s1605 = ((D40[21]*t2130) + (D40[20]*t2131));
            s1606 = ((D40[22]*t2132) - (D40[23]*t2133));
            s1607 = ((D40[23]*t2132) + (D40[22]*t2133));
            t2134 = (s1578 + s1594);
            t2135 = (s1579 + s1595);
            t2136 = (s1562 + t2134);
            t2137 = (s1563 + t2135);
            t2138 = (s1562 - (0.5*t2134));
            t2139 = (s1563 - (0.5*t2135));
            s1608 = (0.8660254037844386*(s1579 - s1595));
            s1609 = (0.8660254037844386*(s1578 - s1594));
            t2140 = (t2138 + s1608);
            t2141 = (t2139 - s1609);
            t2142 = (t2138 - s1608);
            t2143 = (t2139 + s1609);
            t2144 = (s1586 + s1602);
            t2145 = (s1587 + s1603);
            t2146 = (s1570 + t2144);
            t2147 = (s1571 + t2145);
            t2148 = (s1570 - (0.5*t2144));
            t2149 = (s1571 - (0.5*t2145));
            s1610 = (0.8660254037844386*(s1587 - s1603));
            s1611 = (0.8660254037844386*(s1586 - s1602));
            t2150 = (t2148 + s1610);
            t2151 = (t2149 - s1611);
            t2152 = (t2148 - s1610);
            t2153 = (t2149 + s1611);
            s1612 = ((0.5*t2150) + (0.8660254037844386*t2151));
            s1613 = ((0.5*t2151) - (0.8660254037844386*t2150));
            s1614 = ((0.8660254037844386*t2153) - (0.5*t2152));
            s1615 = ((0.8660254037844386*t2152) + (0.5*t2153));
            s1616 = (t2136 - t2146);
            s1617 = (t2137 - t2147);
            s1618 = (t2140 + s1612);
            s1619 = (t2141 + s1613);
            s1620 = (t2140 - s1612);
            s1621 = (t2141 - s1613);
            s1622 = (t2142 + s1614);
            s1623 = (t2143 - s1615);
            s1624 = (t2142 - s1614);
            s1625 = (t2143 + s1615);
            t2154 = (s1580 + s1596);
            t2155 = (s1581 + s1597);
            t2156 = (s1564 + t2154);
            t2157 = (s1565 + t2155);
            t2158 = (s1564 - (0.5*t2154));
            t2159 = (s1565 - (0.5*t2155));
            s1626 = (0.8660254037844386*(s1581 - s1597));
            s1627 = (0.8660254037844386*(s1580 - s1596));
            t2160 = (t2158 + s1626);
            t2161 = (t2159 - s1627);
            t2162 = (t2158 - s1626);
            t2163 = (t2159 + s1627);
            t2164 = (s1588 + s1604);
            t2165 = (s1589 + s1605);
            t2166 = (s1572 + t2164);
            t2167 = (s1573 + t2165);
            t2168 = (s1572 - (0.5*t2164));
            t2169 = (s1573 - (0.5*t2165));
            s1628 = (0.8660254037844386*(s1589 - s1605));
            s1629 = (0.8660254037844386*(s1588 - s1604));
            t2170 = (t2168 + s1628);
            t2171 = (t2169 - s1629);
            t2172 = (t2168 - s1628);
            t2173 = (t2169 + s1629);
            s1630 = ((0.5*t2170) + (0.8660254037844386*t2171));
            s1631 = ((0.5*t2171) - (0.8660254037844386*t2170));
            s1632 = ((0.8660254037844386*t2173) - (0.5*t2172));
            s1633 = ((0.8660254037844386*t2172) + (0.5*t2173));
            s1634 = (t2156 + t2166);
            s1635 = (t2157 + t2167);
            s1636 = (t2156 - t2166);
            s1637 = (t2157 - t2167);
            s1638 = (t2160 + s1630);
            s1639 = (t2161 + s1631);
            s1640 = (t2160 - s1630);
            s1641 = (t2161 - s1631);
            s1642 = (t2162 + s1632);
            s1643 = (t2163 - s1633);
            s1644 = (t2162 - s1632);
            s1645 = (t2163 + s1633);
            t2174 = (s1582 + s1598);
            t2175 = (s1583 + s1599);
            t2176 = (s1566 + t2174);
            t2177 = (s1567 + t2175);
            t2178 = (s1566 - (0.5*t2174));
            t2179 = (s1567 - (0.5*t2175));
            s1646 = (0.8660254037844386*(s1583 - s1599));
            s1647 = (0.8660254037844386*(s1582 - s1598));
            t2180 = (t2178 + s1646);
            t2181 = (t2179 - s1647);
            t2182 = (t2178 - s1646);
            t2183 = (t2179 + s1647);
            t2184 = (s1590 + s1606);
            t2185 = (s1591 + s1607);
            t2186 = (s1574 + t2184);
            t2187 = (s1575 + t2185);
            t2188 = (s1574 - (0.5*t2184));
            t2189 = (s1575 - (0.5*t2185));
            s1648 = (0.8660254037844386*(s1591 - s1607));
            s1649 = (0.8660254037844386*(s1590 - s1606));
            t2190 = (t2188 + s1648);
            t2191 = (t2189 - s1649);
            t2192 = (t2188 - s1648);
            t2193 = (t2189 + s1649);
            s1650 = ((0.5*t2190) + (0.8660254037844386*t2191));
            s1651 = ((0.5*t2191) - (0.8660254037844386*t2190));
            s1652 = ((0.8660254037844386*t2193) - (0.5*t2192));
            s1653 = ((0.8660254037844386*t2192) + (0.5*t2193));
            s1654 = (t2176 + t2186);
            s1655 = (t2177 + t2187);
            s1656 = (t2176 - t2186);
            s1657 = (t2177 - t2187);
            s1658 = (t2180 + s1650);
            s1659 = (t2181 + s1651);
            s1660 = (t2180 - s1650);
            s1661 = (t2181 - s1651);
            s1662 = (t2182 + s1652);
            s1663 = (t2183 - s1653);
            s1664 = (t2182 - s1652);
            s1665 = (t2183 + s1653);
            a3988 = (34*j1);
            a3989 = TW1[a3988];
            a3990 = TW1[(a3988 + 1)];
            a3991 = TW1[(a3988 + 2)];
            a3992 = TW1[(a3988 + 3)];
            a3993 = TW1[(a3988 + 4)];
            a3994 = TW1[(a3988 + 5)];
            a3995 = TW1[(a3988 + 6)];
            a3996 = TW1[(a3988 + 7)];
            a3997 = TW1[(a3988 + 8)];
            a3998 = TW1[(a3988 + 9)];
            a3999 = TW1[(a3988 + 10)];
            a4000 = TW1[(a3988 + 11)];
            a4001 = TW1[(a3988 + 12)];
            a4002 = TW1[(a3988 + 13)];
            a4003 = TW1[(a3988 + 14)];
            a4004 = TW1[(a3988 + 15)];
            a4005 = TW1[(a3988 + 16)];
            a4006 = TW1[(a3988 + 17)];
            a4007 = TW1[(a3988 + 18)];
            a4008 = TW1[(a3988 + 19)];
            a4009 = TW1[(a3988 + 20)];
            a4010 = TW1[(a3988 + 21)];
            a4011 = TW1[(a3988 + 22)];
            a4012 = TW1[(a3988 + 23)];
            a4013 = TW1[(a3988 + 24)];
            a4014 = TW1[(a3988 + 25)];
            a4015 = TW1[(a3988 + 26)];
            a4016 = TW1[(a3988 + 27)];
            a4017 = TW1[(a3988 + 28)];
            a4018 = TW1[(a3988 + 29)];
            a4019 = TW1[(a3988 + 30)];
            a4020 = TW1[(a3988 + 31)];
            a4021 = TW1[(a3988 + 32)];
            a4022 = TW1[(a3988 + 33)];
            a4023 = (36*a3969);
            a4024 = (a3968 + a4023);
            Y[a4024] = (t2136 + t2146);
            Y[(a4024 + 1)] = (t2137 + t2147);
            a4025 = (a3968 + (2*m1) + a4023);
            Y[a4025] = ((a3989*s1634) - (a3990*s1635));
            Y[(a4025 + 1)] = ((a3990*s1634) + (a3989*s1635));
            a4026 = (a3968 + (4*m1) + a4023);
            Y[a4026] = ((a3991*s1654) - (a3992*s1655));
            Y[(a4026 + 1)] = ((a3992*s1654) + (a3991*s1655));
            a4027 = (a3968 + (6*m1) + a4023);
            Y[a4027] = ((a3993*s1618) - (a3994*s1619));
            Y[(a4027 + 1)] = ((a3994*s1618) + (a3993*s1619));
            a4028 = (a3968 + (8*m1) + a4023);
            Y[a4028] = ((a3995*s1638) - (a3996*s1639));
            Y[(a4028 + 1)] = ((a3996*s1638) + (a3995*s1639));
            a4029 = (a3968 + (10*m1) + a4023);
            Y[a4029] = ((a3997*s1658) - (a3998*s1659));
            Y[(a4029 + 1)] = ((a3998*s1658) + (a3997*s1659));
            a4030 = (a3968 + (12*m1) + a4023);
            Y[a4030] = ((a3999*s1622) - (a4000*s1623));
            Y[(a4030 + 1)] = ((a4000*s1622) + (a3999*s1623));
            a4031 = (a3968 + (14*m1) + a4023);
            Y[a4031] = ((a4001*s1642) - (a4002*s1643));
            Y[(a4031 + 1)] = ((a4002*s1642) + (a4001*s1643));
            a4032 = (a3968 + (16*m1) + a4023);
            Y[a4032] = ((a4003*s1662) - (a4004*s1663));
            Y[(a4032 + 1)] = ((a4004*s1662) + (a4003*s1663));
            a4033 = (a3968 + (18*m1) + a4023);
            Y[a4033] = ((a4005*s1616) - (a4006*s1617));
            Y[(a4033 + 1)] = ((a4006*s1616) + (a4005*s1617));
            a4034 = (a3968 + (20*m1) + a4023);
            Y[a4034] = ((a4007*s1636) - (a4008*s1637));
            Y[(a4034 + 1)] = ((a4008*s1636) + (a4007*s1637));
            a4035 = (a3968 + (22*m1) + a4023);
            Y[a4035] = ((a4009*s1656) - (a4010*s1657));
            Y[(a4035 + 1)] = ((a4010*s1656) + (a4009*s1657));
            a4036 = (a3968 + (24*m1) + a4023);
            Y[a4036] = ((a4011*s1620) - (a4012*s1621));
            Y[(a4036 + 1)] = ((a4012*s1620) + (a4011*s1621));
            a4037 = (a3968 + (26*m1) + a4023);
            Y[a4037] = ((a4013*s1640) - (a4014*s1641));
            Y[(a4037 + 1)] = ((a4014*s1640) + (a4013*s1641));
            a4038 = (a3968 + (28*m1) + a4023);
            Y[a4038] = ((a4015*s1660) - (a4016*s1661));
            Y[(a4038 + 1)] = ((a4016*s1660) + (a4015*s1661));
            a4039 = (a3968 + (30*m1) + a4023);
            Y[a4039] = ((a4017*s1624) - (a4018*s1625));
            Y[(a4039 + 1)] = ((a4018*s1624) + (a4017*s1625));
            a4040 = (a3968 + (32*m1) + a4023);
            Y[a4040] = ((a4019*s1644) - (a4020*s1645));
            Y[(a4040 + 1)] = ((a4020*s1644) + (a4019*s1645));
            a4041 = (a3968 + (34*m1) + a4023);
            Y[a4041] = ((a4021*s1664) - (a4022*s1665));
            Y[(a4041 + 1)] = ((a4022*s1664) + (a4021*s1665));
        }
    }
}
