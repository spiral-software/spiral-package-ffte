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

void dft25c_(float64_t  *Y, float64_t  *X, int  *lp1, int  *mp1) {
    static float64_t D11[10];
    static float64_t D12[40];
    int l1, m1;
    svfloat64x2_t s1153, s1156, s1159, s1162, s1165, s1168, s1171, s1174, 
            s1177, s1180, s1183, s1186, s1189, s1192, s1195, s1198, 
            s1201, s1204, s1207, s1210, s1213, s1216, s1219, s1222, 
            s1225, svex2_26, svex2_27, svex2_28, svex2_29, svex2_30, svex2_31, svex2_32, 
            svex2_33, svex2_34, svex2_35, svex2_36, svex2_37, svex2_38, svex2_39, svex2_40, 
            svex2_41, svex2_42, svex2_43, svex2_44, svex2_45, svex2_46, svex2_47, svex2_48, 
            svex2_49, svex2_50;
    svfloat64_t s1154, s1155, s1157, s1158, s1160, s1161, s1163, s1164, 
            s1166, s1167, s1169, s1170, s1172, s1173, s1175, s1176, 
            s1178, s1179, s1181, s1182, s1184, s1185, s1187, s1188, 
            s1190, s1191, s1193, s1194, s1196, s1197, s1199, s1200, 
            s1202, s1203, s1205, s1206, s1208, s1209, s1211, s1212, 
            s1214, s1215, s1217, s1218, s1220, s1221, s1223, s1224, 
            s1226, s1227, s1228, s1229, s1230, s1231, s1232, s1233, 
            s1234, s1235, s1236, s1237, s1238, s1239, s1240, s1241, 
            s1242, s1243, s1244, s1245, s1246, s1247, s1248, s1249, 
            s1250, s1251, s1252, s1253, s1254, s1255, s1256, s1257, 
            s1258, s1259, s1260, s1261, s1262, s1263, s1264, s1265, 
            s1266, s1267, s1268, s1269, s1270, s1271, s1272, s1273, 
            s1274, s1275, s1276, s1277, s1278, s1279, s1280, s1281, 
            s1282, s1283, s1284, s1285, s1286, s1287, s1288, s1289, 
            s1290, s1291, s1292, s1293, s1294, s1295, s1296, s1297, 
            s1298, s1299, s1300, s1301, s1302, s1303, s1304, s1305, 
            s1306, s1307, s1308, s1309, s1310, s1311, s1312, s1313, 
            s1314, s1315, s1316, s1317, s1318, s1319, s1320, s1321, 
            s1322, s1323, s1324, s1325, s1326, s1327, s1328, s1329, 
            s1330, s1331, s1332, s1333, s1334, s1335, s1336, s1337, 
            s1338, s1339, s1340, s1341, s1342, s1343, s1344, s1345, 
            s1346, s1347, s1348, s1349, s1350, s1351, s1352, s1353, 
            s1354, s1355, s1356, s1357, s1358, s1359, s1360, s1361, 
            s1362, s1363, s1364, s1365, s1366, s1367, s1368, s1369, 
            s1370, s1371, s1372, s1373, s1374, s1375, s1376, s1377, 
            t1552, t1553, t1554, t1555, t1556, t1557, t1558, t1559, 
            t1560, t1561, t1562, t1563, t1564, t1565, t1566, t1567, 
            t1568, t1569, t1570, t1571, t1572, t1573, t1574, t1575, 
            t1576, t1577, t1578, t1579, t1580, t1581, t1582, t1583, 
            t1584, t1585, t1586, t1587, t1588, t1589, t1590, t1591, 
            t1592, t1593, t1594, t1595, t1596, t1597, t1598, t1599, 
            t1600, t1601, t1602, t1603, t1604, t1605, t1606, t1607, 
            t1608, t1609, t1610, t1611, t1612, t1613, t1614, t1615, 
            t1616, t1617, t1618, t1619, t1620, t1621, t1622, t1623, 
            t1624, t1625, t1626, t1627, t1628, t1629, t1630, t1631, 
            t1632, t1633, t1634, t1635, t1636, t1637, t1638, t1639, 
            t1640, t1641, t1642, t1643, t1644, t1645, t1646, t1647, 
            t1648, t1649, t1650, t1651, t1652, t1653, t1654, t1655, 
            t1656, t1657, t1658, t1659, t1660, t1661, t1662, t1663, 
            t1664, t1665, t1666, t1667, t1668, t1669, t1670, t1671, 
            t1672, t1673, t1674, t1675, t1676, t1677, t1678, t1679, 
            t1680, t1681, t1682, t1683, t1684, t1685, t1686, t1687, 
            t1688, t1689, t1690, t1691, t1692, t1693, t1694, t1695, 
            t1696, t1697, t1698, t1699, t1700, t1701, t1702, t1703, 
            t1704, t1705, t1706, t1707, t1708, t1709, t1710, t1711, 
            t1712, t1713, t1714, t1715, t1716, t1717, t1718, t1719, 
            t1720, t1721, t1722, t1723, t1724, t1725, t1726, t1727, 
            t1728, t1729, t1730, t1731, t1732, t1733, t1734, t1735, 
            t1736, t1737, t1738, t1739, t1740, t1741, t1742, t1743, 
            t1744, t1745, t1746, t1747, t1748, t1749, t1750, t1751, 
            t1752, t1753, t1754, t1755, t1756, t1757, t1758, t1759, 
            t1760, t1761, t1762, t1763, t1764, t1765, t1766, t1767, 
            t1768, t1769, t1770, t1771, t1772, t1773, t1774, t1775, 
            t1776, t1777, t1778, t1779, t1780, t1781, t1782, t1783, 
            t1784, t1785, t1786, t1787, t1788, t1789, t1790, t1791, 
            t1792, t1793, t1794, t1795, t1796, t1797, t1798, t1799, 
            t1800, t1801, t1802, t1803, t1804, t1805, t1806, t1807, 
            t1808, t1809, t1810, t1811, t1812, t1813, t1814, t1815, 
            t1816, t1817, t1818, t1819, t1820, t1821, t1822, t1823, 
            t1824, t1825, t1826, t1827, t1828, t1829, t1830, t1831, 
            t1832, t1833, t1834, t1835, t1836, t1837, t1838, t1839, 
            t1840, t1841;
    svbool_t pg1;
    l1 = *(lp1);
    m1 = *(mp1);
    int k1 = 0;
    pg1 = svwhilelt_b64(k1, m1);
    do {
        s1153 = svld2_f64(pg1, (X + ((2)*(k1))));
        s1154 = s1153.v0;
        s1155 = s1153.v1;
        s1156 = svld2_f64(pg1, (X + ((2)*((k1 + ((l1)*(m1)))))));
        s1157 = s1156.v0;
        s1158 = s1156.v1;
        s1159 = svld2_f64(pg1, (X + ((2)*((k1 + ((((2)*(l1)))*(m1)))))));
        s1160 = s1159.v0;
        s1161 = s1159.v1;
        s1162 = svld2_f64(pg1, (X + ((2)*((k1 + ((((3)*(l1)))*(m1)))))));
        s1163 = s1162.v0;
        s1164 = s1162.v1;
        s1165 = svld2_f64(pg1, (X + ((2)*((k1 + ((((4)*(l1)))*(m1)))))));
        s1166 = s1165.v0;
        s1167 = s1165.v1;
        s1168 = svld2_f64(pg1, (X + ((2)*((k1 + ((((5)*(l1)))*(m1)))))));
        s1169 = s1168.v0;
        s1170 = s1168.v1;
        s1171 = svld2_f64(pg1, (X + ((2)*((k1 + ((((6)*(l1)))*(m1)))))));
        s1172 = s1171.v0;
        s1173 = s1171.v1;
        s1174 = svld2_f64(pg1, (X + ((2)*((k1 + ((((7)*(l1)))*(m1)))))));
        s1175 = s1174.v0;
        s1176 = s1174.v1;
        s1177 = svld2_f64(pg1, (X + ((2)*((k1 + ((((8)*(l1)))*(m1)))))));
        s1178 = s1177.v0;
        s1179 = s1177.v1;
        s1180 = svld2_f64(pg1, (X + ((2)*((k1 + ((((9)*(l1)))*(m1)))))));
        s1181 = s1180.v0;
        s1182 = s1180.v1;
        s1183 = svld2_f64(pg1, (X + ((2)*((k1 + ((((10)*(l1)))*(m1)))))));
        s1184 = s1183.v0;
        s1185 = s1183.v1;
        s1186 = svld2_f64(pg1, (X + ((2)*((k1 + ((((11)*(l1)))*(m1)))))));
        s1187 = s1186.v0;
        s1188 = s1186.v1;
        s1189 = svld2_f64(pg1, (X + ((2)*((k1 + ((((12)*(l1)))*(m1)))))));
        s1190 = s1189.v0;
        s1191 = s1189.v1;
        s1192 = svld2_f64(pg1, (X + ((2)*((k1 + ((((13)*(l1)))*(m1)))))));
        s1193 = s1192.v0;
        s1194 = s1192.v1;
        s1195 = svld2_f64(pg1, (X + ((2)*((k1 + ((((14)*(l1)))*(m1)))))));
        s1196 = s1195.v0;
        s1197 = s1195.v1;
        s1198 = svld2_f64(pg1, (X + ((2)*((k1 + ((((15)*(l1)))*(m1)))))));
        s1199 = s1198.v0;
        s1200 = s1198.v1;
        s1201 = svld2_f64(pg1, (X + ((2)*((k1 + ((((16)*(l1)))*(m1)))))));
        s1202 = s1201.v0;
        s1203 = s1201.v1;
        s1204 = svld2_f64(pg1, (X + ((2)*((k1 + ((((17)*(l1)))*(m1)))))));
        s1205 = s1204.v0;
        s1206 = s1204.v1;
        s1207 = svld2_f64(pg1, (X + ((2)*((k1 + ((((18)*(l1)))*(m1)))))));
        s1208 = s1207.v0;
        s1209 = s1207.v1;
        s1210 = svld2_f64(pg1, (X + ((2)*((k1 + ((((19)*(l1)))*(m1)))))));
        s1211 = s1210.v0;
        s1212 = s1210.v1;
        s1213 = svld2_f64(pg1, (X + ((2)*((k1 + ((((20)*(l1)))*(m1)))))));
        s1214 = s1213.v0;
        s1215 = s1213.v1;
        s1216 = svld2_f64(pg1, (X + ((2)*((k1 + ((((21)*(l1)))*(m1)))))));
        s1217 = s1216.v0;
        s1218 = s1216.v1;
        s1219 = svld2_f64(pg1, (X + ((2)*((k1 + ((((22)*(l1)))*(m1)))))));
        s1220 = s1219.v0;
        s1221 = s1219.v1;
        s1222 = svld2_f64(pg1, (X + ((2)*((k1 + ((((23)*(l1)))*(m1)))))));
        s1223 = s1222.v0;
        s1224 = s1222.v1;
        s1225 = svld2_f64(pg1, (X + ((2)*((k1 + ((((24)*(l1)))*(m1)))))));
        s1226 = s1225.v0;
        s1227 = s1225.v1;
        t1552 = svadd_f64_x(pg1, s1169, s1214);
        t1553 = svadd_f64_x(pg1, s1170, s1215);
        t1554 = svsub_f64_x(pg1, s1169, s1214);
        t1555 = svsub_f64_x(pg1, s1170, s1215);
        t1556 = svadd_f64_x(pg1, s1184, s1199);
        t1557 = svadd_f64_x(pg1, s1185, s1200);
        t1558 = svsub_f64_x(pg1, s1184, s1199);
        t1559 = svsub_f64_x(pg1, s1185, s1200);
        t1560 = svadd_f64_x(pg1, t1552, t1556);
        t1561 = svadd_f64_x(pg1, t1553, t1557);
        t1562 = svadd_f64_x(pg1, t1554, t1559);
        t1563 = svsub_f64_x(pg1, t1555, t1558);
        t1564 = svsub_f64_x(pg1, t1554, t1559);
        t1565 = svadd_f64_x(pg1, t1555, t1558);
        t1566 = svadd_f64_x(pg1, s1154, t1560);
        t1567 = svadd_f64_x(pg1, s1155, t1561);
        t1568 = svmls_n_f64_x(pg1, s1154, t1560, 0.25);
        t1569 = svmls_n_f64_x(pg1, s1155, t1561, 0.25);
        s1338 = svmla_n_f64_x(pg1, t1562, t1563, 1.6180339887498947);
        s1228 = svmul_n_f64_x(pg1, s1338, 0.29389262614623657);
        s1339 = svmls_n_f64_x(pg1, t1563, t1562, 1.6180339887498947);
        s1229 = svmul_n_f64_x(pg1, s1339, 0.29389262614623657);
        s1230 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1552, t1556), 0.55901699437494745);
        s1231 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1553, t1557), 0.55901699437494745);
        s1340 = svmls_n_f64_x(pg1, t1565, t1564, 0.61803398874989479);
        s1232 = svmul_n_f64_x(pg1, s1340, 0.47552825814757682);
        s1341 = svmla_n_f64_x(pg1, t1564, t1565, 0.61803398874989479);
        s1233 = svmul_n_f64_x(pg1, s1341, 0.47552825814757682);
        s1234 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1567, D11[1]), t1566, D11[0]);
        s1235 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1567, D11[0]), t1566, D11[1]);
        t1570 = svadd_f64_x(pg1, t1568, s1230);
        t1571 = svadd_f64_x(pg1, t1569, s1231);
        t1572 = svsub_f64_x(pg1, t1568, s1230);
        t1573 = svsub_f64_x(pg1, t1569, s1231);
        t1574 = svadd_f64_x(pg1, s1228, s1232);
        t1575 = svsub_f64_x(pg1, s1229, s1233);
        t1576 = svsub_f64_x(pg1, s1228, s1232);
        t1577 = svadd_f64_x(pg1, s1229, s1233);
        t1578 = svadd_f64_x(pg1, t1570, t1574);
        t1579 = svadd_f64_x(pg1, t1571, t1575);
        t1580 = svsub_f64_x(pg1, t1570, t1574);
        t1581 = svsub_f64_x(pg1, t1571, t1575);
        s1236 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1579, D12[1]), t1578, D12[0]);
        s1237 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1579, D12[0]), t1578, D12[1]);
        s1238 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1581, D12[3]), t1580, D12[2]);
        s1239 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1581, D12[2]), t1580, D12[3]);
        t1582 = svadd_f64_x(pg1, t1572, t1577);
        t1583 = svsub_f64_x(pg1, t1573, t1576);
        t1584 = svsub_f64_x(pg1, t1572, t1577);
        t1585 = svadd_f64_x(pg1, t1573, t1576);
        s1240 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1583, D12[5]), t1582, D12[4]);
        s1241 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1583, D12[4]), t1582, D12[5]);
        s1242 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1585, D12[7]), t1584, D12[6]);
        s1243 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1585, D12[6]), t1584, D12[7]);
        t1586 = svadd_f64_x(pg1, s1172, s1217);
        t1587 = svadd_f64_x(pg1, s1173, s1218);
        t1588 = svsub_f64_x(pg1, s1172, s1217);
        t1589 = svsub_f64_x(pg1, s1173, s1218);
        t1590 = svadd_f64_x(pg1, s1187, s1202);
        t1591 = svadd_f64_x(pg1, s1188, s1203);
        t1592 = svsub_f64_x(pg1, s1187, s1202);
        t1593 = svsub_f64_x(pg1, s1188, s1203);
        t1594 = svadd_f64_x(pg1, t1586, t1590);
        t1595 = svadd_f64_x(pg1, t1587, t1591);
        t1596 = svadd_f64_x(pg1, t1588, t1593);
        t1597 = svsub_f64_x(pg1, t1589, t1592);
        t1598 = svsub_f64_x(pg1, t1588, t1593);
        t1599 = svadd_f64_x(pg1, t1589, t1592);
        t1600 = svadd_f64_x(pg1, s1157, t1594);
        t1601 = svadd_f64_x(pg1, s1158, t1595);
        t1602 = svmls_n_f64_x(pg1, s1157, t1594, 0.25);
        t1603 = svmls_n_f64_x(pg1, s1158, t1595, 0.25);
        s1342 = svmla_n_f64_x(pg1, t1596, t1597, 1.6180339887498947);
        s1244 = svmul_n_f64_x(pg1, s1342, 0.29389262614623657);
        s1343 = svmls_n_f64_x(pg1, t1597, t1596, 1.6180339887498947);
        s1245 = svmul_n_f64_x(pg1, s1343, 0.29389262614623657);
        s1246 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1586, t1590), 0.55901699437494745);
        s1247 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1587, t1591), 0.55901699437494745);
        s1344 = svmls_n_f64_x(pg1, t1599, t1598, 0.61803398874989479);
        s1248 = svmul_n_f64_x(pg1, s1344, 0.47552825814757682);
        s1345 = svmla_n_f64_x(pg1, t1598, t1599, 0.61803398874989479);
        s1249 = svmul_n_f64_x(pg1, s1345, 0.47552825814757682);
        s1250 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1601, D11[3]), t1600, D11[2]);
        s1251 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1601, D11[2]), t1600, D11[3]);
        t1604 = svadd_f64_x(pg1, t1602, s1246);
        t1605 = svadd_f64_x(pg1, t1603, s1247);
        t1606 = svsub_f64_x(pg1, t1602, s1246);
        t1607 = svsub_f64_x(pg1, t1603, s1247);
        t1608 = svadd_f64_x(pg1, s1244, s1248);
        t1609 = svsub_f64_x(pg1, s1245, s1249);
        t1610 = svsub_f64_x(pg1, s1244, s1248);
        t1611 = svadd_f64_x(pg1, s1245, s1249);
        t1612 = svadd_f64_x(pg1, t1604, t1608);
        t1613 = svadd_f64_x(pg1, t1605, t1609);
        t1614 = svsub_f64_x(pg1, t1604, t1608);
        t1615 = svsub_f64_x(pg1, t1605, t1609);
        s1252 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1613, D12[9]), t1612, D12[8]);
        s1253 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1613, D12[8]), t1612, D12[9]);
        s1254 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1615, D12[11]), t1614, D12[10]);
        s1255 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1615, D12[10]), t1614, D12[11]);
        t1616 = svadd_f64_x(pg1, t1606, t1611);
        t1617 = svsub_f64_x(pg1, t1607, t1610);
        t1618 = svsub_f64_x(pg1, t1606, t1611);
        t1619 = svadd_f64_x(pg1, t1607, t1610);
        s1256 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1617, D12[13]), t1616, D12[12]);
        s1257 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1617, D12[12]), t1616, D12[13]);
        s1258 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1619, D12[15]), t1618, D12[14]);
        s1259 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1619, D12[14]), t1618, D12[15]);
        t1620 = svadd_f64_x(pg1, s1175, s1220);
        t1621 = svadd_f64_x(pg1, s1176, s1221);
        t1622 = svsub_f64_x(pg1, s1175, s1220);
        t1623 = svsub_f64_x(pg1, s1176, s1221);
        t1624 = svadd_f64_x(pg1, s1190, s1205);
        t1625 = svadd_f64_x(pg1, s1191, s1206);
        t1626 = svsub_f64_x(pg1, s1190, s1205);
        t1627 = svsub_f64_x(pg1, s1191, s1206);
        t1628 = svadd_f64_x(pg1, t1620, t1624);
        t1629 = svadd_f64_x(pg1, t1621, t1625);
        t1630 = svadd_f64_x(pg1, t1622, t1627);
        t1631 = svsub_f64_x(pg1, t1623, t1626);
        t1632 = svsub_f64_x(pg1, t1622, t1627);
        t1633 = svadd_f64_x(pg1, t1623, t1626);
        t1634 = svadd_f64_x(pg1, s1160, t1628);
        t1635 = svadd_f64_x(pg1, s1161, t1629);
        t1636 = svmls_n_f64_x(pg1, s1160, t1628, 0.25);
        t1637 = svmls_n_f64_x(pg1, s1161, t1629, 0.25);
        s1346 = svmla_n_f64_x(pg1, t1630, t1631, 1.6180339887498947);
        s1260 = svmul_n_f64_x(pg1, s1346, 0.29389262614623657);
        s1347 = svmls_n_f64_x(pg1, t1631, t1630, 1.6180339887498947);
        s1261 = svmul_n_f64_x(pg1, s1347, 0.29389262614623657);
        s1262 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1620, t1624), 0.55901699437494745);
        s1263 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1621, t1625), 0.55901699437494745);
        s1348 = svmls_n_f64_x(pg1, t1633, t1632, 0.61803398874989479);
        s1264 = svmul_n_f64_x(pg1, s1348, 0.47552825814757682);
        s1349 = svmla_n_f64_x(pg1, t1632, t1633, 0.61803398874989479);
        s1265 = svmul_n_f64_x(pg1, s1349, 0.47552825814757682);
        s1266 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1635, D11[5]), t1634, D11[4]);
        s1267 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1635, D11[4]), t1634, D11[5]);
        t1638 = svadd_f64_x(pg1, t1636, s1262);
        t1639 = svadd_f64_x(pg1, t1637, s1263);
        t1640 = svsub_f64_x(pg1, t1636, s1262);
        t1641 = svsub_f64_x(pg1, t1637, s1263);
        t1642 = svadd_f64_x(pg1, s1260, s1264);
        t1643 = svsub_f64_x(pg1, s1261, s1265);
        t1644 = svsub_f64_x(pg1, s1260, s1264);
        t1645 = svadd_f64_x(pg1, s1261, s1265);
        t1646 = svadd_f64_x(pg1, t1638, t1642);
        t1647 = svadd_f64_x(pg1, t1639, t1643);
        t1648 = svsub_f64_x(pg1, t1638, t1642);
        t1649 = svsub_f64_x(pg1, t1639, t1643);
        s1268 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1647, D12[17]), t1646, D12[16]);
        s1269 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1647, D12[16]), t1646, D12[17]);
        s1270 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1649, D12[19]), t1648, D12[18]);
        s1271 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1649, D12[18]), t1648, D12[19]);
        t1650 = svadd_f64_x(pg1, t1640, t1645);
        t1651 = svsub_f64_x(pg1, t1641, t1644);
        t1652 = svsub_f64_x(pg1, t1640, t1645);
        t1653 = svadd_f64_x(pg1, t1641, t1644);
        s1272 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1651, D12[21]), t1650, D12[20]);
        s1273 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1651, D12[20]), t1650, D12[21]);
        s1274 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1653, D12[23]), t1652, D12[22]);
        s1275 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1653, D12[22]), t1652, D12[23]);
        t1654 = svadd_f64_x(pg1, s1178, s1223);
        t1655 = svadd_f64_x(pg1, s1179, s1224);
        t1656 = svsub_f64_x(pg1, s1178, s1223);
        t1657 = svsub_f64_x(pg1, s1179, s1224);
        t1658 = svadd_f64_x(pg1, s1193, s1208);
        t1659 = svadd_f64_x(pg1, s1194, s1209);
        t1660 = svsub_f64_x(pg1, s1193, s1208);
        t1661 = svsub_f64_x(pg1, s1194, s1209);
        t1662 = svadd_f64_x(pg1, t1654, t1658);
        t1663 = svadd_f64_x(pg1, t1655, t1659);
        t1664 = svadd_f64_x(pg1, t1656, t1661);
        t1665 = svsub_f64_x(pg1, t1657, t1660);
        t1666 = svsub_f64_x(pg1, t1656, t1661);
        t1667 = svadd_f64_x(pg1, t1657, t1660);
        t1668 = svadd_f64_x(pg1, s1163, t1662);
        t1669 = svadd_f64_x(pg1, s1164, t1663);
        t1670 = svmls_n_f64_x(pg1, s1163, t1662, 0.25);
        t1671 = svmls_n_f64_x(pg1, s1164, t1663, 0.25);
        s1350 = svmla_n_f64_x(pg1, t1664, t1665, 1.6180339887498947);
        s1276 = svmul_n_f64_x(pg1, s1350, 0.29389262614623657);
        s1351 = svmls_n_f64_x(pg1, t1665, t1664, 1.6180339887498947);
        s1277 = svmul_n_f64_x(pg1, s1351, 0.29389262614623657);
        s1278 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1654, t1658), 0.55901699437494745);
        s1279 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1655, t1659), 0.55901699437494745);
        s1352 = svmls_n_f64_x(pg1, t1667, t1666, 0.61803398874989479);
        s1280 = svmul_n_f64_x(pg1, s1352, 0.47552825814757682);
        s1353 = svmla_n_f64_x(pg1, t1666, t1667, 0.61803398874989479);
        s1281 = svmul_n_f64_x(pg1, s1353, 0.47552825814757682);
        s1282 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1669, D11[7]), t1668, D11[6]);
        s1283 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1669, D11[6]), t1668, D11[7]);
        t1672 = svadd_f64_x(pg1, t1670, s1278);
        t1673 = svadd_f64_x(pg1, t1671, s1279);
        t1674 = svsub_f64_x(pg1, t1670, s1278);
        t1675 = svsub_f64_x(pg1, t1671, s1279);
        t1676 = svadd_f64_x(pg1, s1276, s1280);
        t1677 = svsub_f64_x(pg1, s1277, s1281);
        t1678 = svsub_f64_x(pg1, s1276, s1280);
        t1679 = svadd_f64_x(pg1, s1277, s1281);
        t1680 = svadd_f64_x(pg1, t1672, t1676);
        t1681 = svadd_f64_x(pg1, t1673, t1677);
        t1682 = svsub_f64_x(pg1, t1672, t1676);
        t1683 = svsub_f64_x(pg1, t1673, t1677);
        s1284 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1681, D12[25]), t1680, D12[24]);
        s1285 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1681, D12[24]), t1680, D12[25]);
        s1286 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1683, D12[27]), t1682, D12[26]);
        s1287 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1683, D12[26]), t1682, D12[27]);
        t1684 = svadd_f64_x(pg1, t1674, t1679);
        t1685 = svsub_f64_x(pg1, t1675, t1678);
        t1686 = svsub_f64_x(pg1, t1674, t1679);
        t1687 = svadd_f64_x(pg1, t1675, t1678);
        s1288 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1685, D12[29]), t1684, D12[28]);
        s1289 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1685, D12[28]), t1684, D12[29]);
        s1290 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1687, D12[31]), t1686, D12[30]);
        s1291 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1687, D12[30]), t1686, D12[31]);
        t1688 = svadd_f64_x(pg1, s1181, s1226);
        t1689 = svadd_f64_x(pg1, s1182, s1227);
        t1690 = svsub_f64_x(pg1, s1181, s1226);
        t1691 = svsub_f64_x(pg1, s1182, s1227);
        t1692 = svadd_f64_x(pg1, s1196, s1211);
        t1693 = svadd_f64_x(pg1, s1197, s1212);
        t1694 = svsub_f64_x(pg1, s1196, s1211);
        t1695 = svsub_f64_x(pg1, s1197, s1212);
        t1696 = svadd_f64_x(pg1, t1688, t1692);
        t1697 = svadd_f64_x(pg1, t1689, t1693);
        t1698 = svadd_f64_x(pg1, t1690, t1695);
        t1699 = svsub_f64_x(pg1, t1691, t1694);
        t1700 = svsub_f64_x(pg1, t1690, t1695);
        t1701 = svadd_f64_x(pg1, t1691, t1694);
        t1702 = svadd_f64_x(pg1, s1166, t1696);
        t1703 = svadd_f64_x(pg1, s1167, t1697);
        t1704 = svmls_n_f64_x(pg1, s1166, t1696, 0.25);
        t1705 = svmls_n_f64_x(pg1, s1167, t1697, 0.25);
        s1354 = svmla_n_f64_x(pg1, t1698, t1699, 1.6180339887498947);
        s1292 = svmul_n_f64_x(pg1, s1354, 0.29389262614623657);
        s1355 = svmls_n_f64_x(pg1, t1699, t1698, 1.6180339887498947);
        s1293 = svmul_n_f64_x(pg1, s1355, 0.29389262614623657);
        s1294 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1688, t1692), 0.55901699437494745);
        s1295 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1689, t1693), 0.55901699437494745);
        s1356 = svmls_n_f64_x(pg1, t1701, t1700, 0.61803398874989479);
        s1296 = svmul_n_f64_x(pg1, s1356, 0.47552825814757682);
        s1357 = svmla_n_f64_x(pg1, t1700, t1701, 0.61803398874989479);
        s1297 = svmul_n_f64_x(pg1, s1357, 0.47552825814757682);
        s1298 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1703, D11[9]), t1702, D11[8]);
        s1299 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1703, D11[8]), t1702, D11[9]);
        t1706 = svadd_f64_x(pg1, t1704, s1294);
        t1707 = svadd_f64_x(pg1, t1705, s1295);
        t1708 = svsub_f64_x(pg1, t1704, s1294);
        t1709 = svsub_f64_x(pg1, t1705, s1295);
        t1710 = svadd_f64_x(pg1, s1292, s1296);
        t1711 = svsub_f64_x(pg1, s1293, s1297);
        t1712 = svsub_f64_x(pg1, s1292, s1296);
        t1713 = svadd_f64_x(pg1, s1293, s1297);
        t1714 = svadd_f64_x(pg1, t1706, t1710);
        t1715 = svadd_f64_x(pg1, t1707, t1711);
        t1716 = svsub_f64_x(pg1, t1706, t1710);
        t1717 = svsub_f64_x(pg1, t1707, t1711);
        s1300 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1715, D12[33]), t1714, D12[32]);
        s1301 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1715, D12[32]), t1714, D12[33]);
        s1302 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1717, D12[35]), t1716, D12[34]);
        s1303 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1717, D12[34]), t1716, D12[35]);
        t1718 = svadd_f64_x(pg1, t1708, t1713);
        t1719 = svsub_f64_x(pg1, t1709, t1712);
        t1720 = svsub_f64_x(pg1, t1708, t1713);
        t1721 = svadd_f64_x(pg1, t1709, t1712);
        s1304 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1719, D12[37]), t1718, D12[36]);
        s1305 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1719, D12[36]), t1718, D12[37]);
        s1306 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t1721, D12[39]), t1720, D12[38]);
        s1307 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t1721, D12[38]), t1720, D12[39]);
        t1722 = svadd_f64_x(pg1, s1250, s1298);
        t1723 = svadd_f64_x(pg1, s1251, s1299);
        t1724 = svsub_f64_x(pg1, s1250, s1298);
        t1725 = svsub_f64_x(pg1, s1251, s1299);
        t1726 = svadd_f64_x(pg1, s1266, s1282);
        t1727 = svadd_f64_x(pg1, s1267, s1283);
        t1728 = svsub_f64_x(pg1, s1266, s1282);
        t1729 = svsub_f64_x(pg1, s1267, s1283);
        t1730 = svadd_f64_x(pg1, t1722, t1726);
        t1731 = svadd_f64_x(pg1, t1723, t1727);
        t1732 = svadd_f64_x(pg1, t1724, t1729);
        t1733 = svsub_f64_x(pg1, t1725, t1728);
        t1734 = svsub_f64_x(pg1, t1724, t1729);
        t1735 = svadd_f64_x(pg1, t1725, t1728);
        t1736 = svmls_n_f64_x(pg1, s1234, t1730, 0.25);
        t1737 = svmls_n_f64_x(pg1, s1235, t1731, 0.25);
        s1358 = svmla_n_f64_x(pg1, t1732, t1733, 1.6180339887498947);
        s1308 = svmul_n_f64_x(pg1, s1358, 0.29389262614623657);
        s1359 = svmls_n_f64_x(pg1, t1733, t1732, 1.6180339887498947);
        s1309 = svmul_n_f64_x(pg1, s1359, 0.29389262614623657);
        s1310 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1722, t1726), 0.55901699437494745);
        s1311 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1723, t1727), 0.55901699437494745);
        s1360 = svmls_n_f64_x(pg1, t1735, t1734, 0.61803398874989479);
        s1312 = svmul_n_f64_x(pg1, s1360, 0.47552825814757682);
        s1361 = svmla_n_f64_x(pg1, t1734, t1735, 0.61803398874989479);
        s1313 = svmul_n_f64_x(pg1, s1361, 0.47552825814757682);
        t1738 = svadd_f64_x(pg1, t1736, s1310);
        t1739 = svadd_f64_x(pg1, t1737, s1311);
        t1740 = svsub_f64_x(pg1, t1736, s1310);
        t1741 = svsub_f64_x(pg1, t1737, s1311);
        t1742 = svadd_f64_x(pg1, s1308, s1312);
        t1743 = svsub_f64_x(pg1, s1309, s1313);
        t1744 = svsub_f64_x(pg1, s1308, s1312);
        t1745 = svadd_f64_x(pg1, s1309, s1313);
        t1746 = svadd_f64_x(pg1, s1252, s1300);
        t1747 = svadd_f64_x(pg1, s1253, s1301);
        t1748 = svsub_f64_x(pg1, s1252, s1300);
        t1749 = svsub_f64_x(pg1, s1253, s1301);
        t1750 = svadd_f64_x(pg1, s1268, s1284);
        t1751 = svadd_f64_x(pg1, s1269, s1285);
        t1752 = svsub_f64_x(pg1, s1268, s1284);
        t1753 = svsub_f64_x(pg1, s1269, s1285);
        t1754 = svadd_f64_x(pg1, t1746, t1750);
        t1755 = svadd_f64_x(pg1, t1747, t1751);
        t1756 = svadd_f64_x(pg1, t1748, t1753);
        t1757 = svsub_f64_x(pg1, t1749, t1752);
        t1758 = svsub_f64_x(pg1, t1748, t1753);
        t1759 = svadd_f64_x(pg1, t1749, t1752);
        t1760 = svmls_n_f64_x(pg1, s1236, t1754, 0.25);
        t1761 = svmls_n_f64_x(pg1, s1237, t1755, 0.25);
        s1362 = svmla_n_f64_x(pg1, t1756, t1757, 1.6180339887498947);
        s1314 = svmul_n_f64_x(pg1, s1362, 0.29389262614623657);
        s1363 = svmls_n_f64_x(pg1, t1757, t1756, 1.6180339887498947);
        s1315 = svmul_n_f64_x(pg1, s1363, 0.29389262614623657);
        s1316 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1746, t1750), 0.55901699437494745);
        s1317 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1747, t1751), 0.55901699437494745);
        s1364 = svmls_n_f64_x(pg1, t1759, t1758, 0.61803398874989479);
        s1318 = svmul_n_f64_x(pg1, s1364, 0.47552825814757682);
        s1365 = svmla_n_f64_x(pg1, t1758, t1759, 0.61803398874989479);
        s1319 = svmul_n_f64_x(pg1, s1365, 0.47552825814757682);
        t1762 = svadd_f64_x(pg1, t1760, s1316);
        t1763 = svadd_f64_x(pg1, t1761, s1317);
        t1764 = svsub_f64_x(pg1, t1760, s1316);
        t1765 = svsub_f64_x(pg1, t1761, s1317);
        t1766 = svadd_f64_x(pg1, s1314, s1318);
        t1767 = svsub_f64_x(pg1, s1315, s1319);
        t1768 = svsub_f64_x(pg1, s1314, s1318);
        t1769 = svadd_f64_x(pg1, s1315, s1319);
        t1770 = svadd_f64_x(pg1, s1256, s1304);
        t1771 = svadd_f64_x(pg1, s1257, s1305);
        t1772 = svsub_f64_x(pg1, s1256, s1304);
        t1773 = svsub_f64_x(pg1, s1257, s1305);
        t1774 = svadd_f64_x(pg1, s1272, s1288);
        t1775 = svadd_f64_x(pg1, s1273, s1289);
        t1776 = svsub_f64_x(pg1, s1272, s1288);
        t1777 = svsub_f64_x(pg1, s1273, s1289);
        t1778 = svadd_f64_x(pg1, t1770, t1774);
        t1779 = svadd_f64_x(pg1, t1771, t1775);
        t1780 = svadd_f64_x(pg1, t1772, t1777);
        t1781 = svsub_f64_x(pg1, t1773, t1776);
        t1782 = svsub_f64_x(pg1, t1772, t1777);
        t1783 = svadd_f64_x(pg1, t1773, t1776);
        t1784 = svmls_n_f64_x(pg1, s1240, t1778, 0.25);
        t1785 = svmls_n_f64_x(pg1, s1241, t1779, 0.25);
        s1366 = svmla_n_f64_x(pg1, t1780, t1781, 1.6180339887498947);
        s1320 = svmul_n_f64_x(pg1, s1366, 0.29389262614623657);
        s1367 = svmls_n_f64_x(pg1, t1781, t1780, 1.6180339887498947);
        s1321 = svmul_n_f64_x(pg1, s1367, 0.29389262614623657);
        s1322 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1770, t1774), 0.55901699437494745);
        s1323 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1771, t1775), 0.55901699437494745);
        s1368 = svmls_n_f64_x(pg1, t1783, t1782, 0.61803398874989479);
        s1324 = svmul_n_f64_x(pg1, s1368, 0.47552825814757682);
        s1369 = svmla_n_f64_x(pg1, t1782, t1783, 0.61803398874989479);
        s1325 = svmul_n_f64_x(pg1, s1369, 0.47552825814757682);
        t1786 = svadd_f64_x(pg1, t1784, s1322);
        t1787 = svadd_f64_x(pg1, t1785, s1323);
        t1788 = svsub_f64_x(pg1, t1784, s1322);
        t1789 = svsub_f64_x(pg1, t1785, s1323);
        t1790 = svadd_f64_x(pg1, s1320, s1324);
        t1791 = svsub_f64_x(pg1, s1321, s1325);
        t1792 = svsub_f64_x(pg1, s1320, s1324);
        t1793 = svadd_f64_x(pg1, s1321, s1325);
        t1794 = svadd_f64_x(pg1, s1258, s1306);
        t1795 = svadd_f64_x(pg1, s1259, s1307);
        t1796 = svsub_f64_x(pg1, s1258, s1306);
        t1797 = svsub_f64_x(pg1, s1259, s1307);
        t1798 = svadd_f64_x(pg1, s1274, s1290);
        t1799 = svadd_f64_x(pg1, s1275, s1291);
        t1800 = svsub_f64_x(pg1, s1274, s1290);
        t1801 = svsub_f64_x(pg1, s1275, s1291);
        t1802 = svadd_f64_x(pg1, t1794, t1798);
        t1803 = svadd_f64_x(pg1, t1795, t1799);
        t1804 = svadd_f64_x(pg1, t1796, t1801);
        t1805 = svsub_f64_x(pg1, t1797, t1800);
        t1806 = svsub_f64_x(pg1, t1796, t1801);
        t1807 = svadd_f64_x(pg1, t1797, t1800);
        t1808 = svmls_n_f64_x(pg1, s1242, t1802, 0.25);
        t1809 = svmls_n_f64_x(pg1, s1243, t1803, 0.25);
        s1370 = svmla_n_f64_x(pg1, t1804, t1805, 1.6180339887498947);
        s1326 = svmul_n_f64_x(pg1, s1370, 0.29389262614623657);
        s1371 = svmls_n_f64_x(pg1, t1805, t1804, 1.6180339887498947);
        s1327 = svmul_n_f64_x(pg1, s1371, 0.29389262614623657);
        s1328 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1794, t1798), 0.55901699437494745);
        s1329 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1795, t1799), 0.55901699437494745);
        s1372 = svmls_n_f64_x(pg1, t1807, t1806, 0.61803398874989479);
        s1330 = svmul_n_f64_x(pg1, s1372, 0.47552825814757682);
        s1373 = svmla_n_f64_x(pg1, t1806, t1807, 0.61803398874989479);
        s1331 = svmul_n_f64_x(pg1, s1373, 0.47552825814757682);
        t1810 = svadd_f64_x(pg1, t1808, s1328);
        t1811 = svadd_f64_x(pg1, t1809, s1329);
        t1812 = svsub_f64_x(pg1, t1808, s1328);
        t1813 = svsub_f64_x(pg1, t1809, s1329);
        t1814 = svadd_f64_x(pg1, s1326, s1330);
        t1815 = svsub_f64_x(pg1, s1327, s1331);
        t1816 = svsub_f64_x(pg1, s1326, s1330);
        t1817 = svadd_f64_x(pg1, s1327, s1331);
        t1818 = svadd_f64_x(pg1, s1254, s1302);
        t1819 = svadd_f64_x(pg1, s1255, s1303);
        t1820 = svsub_f64_x(pg1, s1254, s1302);
        t1821 = svsub_f64_x(pg1, s1255, s1303);
        t1822 = svadd_f64_x(pg1, s1270, s1286);
        t1823 = svadd_f64_x(pg1, s1271, s1287);
        t1824 = svsub_f64_x(pg1, s1270, s1286);
        t1825 = svsub_f64_x(pg1, s1271, s1287);
        t1826 = svadd_f64_x(pg1, t1818, t1822);
        t1827 = svadd_f64_x(pg1, t1819, t1823);
        t1828 = svadd_f64_x(pg1, t1820, t1825);
        t1829 = svsub_f64_x(pg1, t1821, t1824);
        t1830 = svsub_f64_x(pg1, t1820, t1825);
        t1831 = svadd_f64_x(pg1, t1821, t1824);
        t1832 = svmls_n_f64_x(pg1, s1238, t1826, 0.25);
        t1833 = svmls_n_f64_x(pg1, s1239, t1827, 0.25);
        s1374 = svmla_n_f64_x(pg1, t1828, t1829, 1.6180339887498947);
        s1332 = svmul_n_f64_x(pg1, s1374, 0.29389262614623657);
        s1375 = svmls_n_f64_x(pg1, t1829, t1828, 1.6180339887498947);
        s1333 = svmul_n_f64_x(pg1, s1375, 0.29389262614623657);
        s1334 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1818, t1822), 0.55901699437494745);
        s1335 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, t1819, t1823), 0.55901699437494745);
        s1376 = svmls_n_f64_x(pg1, t1831, t1830, 0.61803398874989479);
        s1336 = svmul_n_f64_x(pg1, s1376, 0.47552825814757682);
        s1377 = svmla_n_f64_x(pg1, t1830, t1831, 0.61803398874989479);
        s1337 = svmul_n_f64_x(pg1, s1377, 0.47552825814757682);
        t1834 = svadd_f64_x(pg1, t1832, s1334);
        t1835 = svadd_f64_x(pg1, t1833, s1335);
        t1836 = svsub_f64_x(pg1, t1832, s1334);
        t1837 = svsub_f64_x(pg1, t1833, s1335);
        t1838 = svadd_f64_x(pg1, s1332, s1336);
        t1839 = svsub_f64_x(pg1, s1333, s1337);
        t1840 = svsub_f64_x(pg1, s1332, s1336);
        t1841 = svadd_f64_x(pg1, s1333, s1337);
        svex2_26.v0 = svadd_f64_x(pg1, s1234, t1730);
        svex2_26.v1 = svadd_f64_x(pg1, s1235, t1731);
        svst2_f64(pg1, (Y + ((2)*(k1))), svex2_26);
        svex2_27.v0 = svadd_f64_x(pg1, s1236, t1754);
        svex2_27.v1 = svadd_f64_x(pg1, s1237, t1755);
        svst2_f64(pg1, (Y + ((2)*((k1 + m1)))), svex2_27);
        svex2_28.v0 = svadd_f64_x(pg1, s1240, t1778);
        svex2_28.v1 = svadd_f64_x(pg1, s1241, t1779);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((2)*(m1)))))), svex2_28);
        svex2_29.v0 = svadd_f64_x(pg1, s1242, t1802);
        svex2_29.v1 = svadd_f64_x(pg1, s1243, t1803);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((3)*(m1)))))), svex2_29);
        svex2_30.v0 = svadd_f64_x(pg1, s1238, t1826);
        svex2_30.v1 = svadd_f64_x(pg1, s1239, t1827);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((4)*(m1)))))), svex2_30);
        svex2_31.v0 = svadd_f64_x(pg1, t1738, t1742);
        svex2_31.v1 = svadd_f64_x(pg1, t1739, t1743);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((5)*(m1)))))), svex2_31);
        svex2_32.v0 = svadd_f64_x(pg1, t1762, t1766);
        svex2_32.v1 = svadd_f64_x(pg1, t1763, t1767);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((6)*(m1)))))), svex2_32);
        svex2_33.v0 = svadd_f64_x(pg1, t1786, t1790);
        svex2_33.v1 = svadd_f64_x(pg1, t1787, t1791);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((7)*(m1)))))), svex2_33);
        svex2_34.v0 = svadd_f64_x(pg1, t1810, t1814);
        svex2_34.v1 = svadd_f64_x(pg1, t1811, t1815);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((8)*(m1)))))), svex2_34);
        svex2_35.v0 = svadd_f64_x(pg1, t1834, t1838);
        svex2_35.v1 = svadd_f64_x(pg1, t1835, t1839);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((9)*(m1)))))), svex2_35);
        svex2_36.v0 = svadd_f64_x(pg1, t1740, t1745);
        svex2_36.v1 = svsub_f64_x(pg1, t1741, t1744);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((10)*(m1)))))), svex2_36);
        svex2_37.v0 = svadd_f64_x(pg1, t1764, t1769);
        svex2_37.v1 = svsub_f64_x(pg1, t1765, t1768);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((11)*(m1)))))), svex2_37);
        svex2_38.v0 = svadd_f64_x(pg1, t1788, t1793);
        svex2_38.v1 = svsub_f64_x(pg1, t1789, t1792);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((12)*(m1)))))), svex2_38);
        svex2_39.v0 = svadd_f64_x(pg1, t1812, t1817);
        svex2_39.v1 = svsub_f64_x(pg1, t1813, t1816);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((13)*(m1)))))), svex2_39);
        svex2_40.v0 = svadd_f64_x(pg1, t1836, t1841);
        svex2_40.v1 = svsub_f64_x(pg1, t1837, t1840);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((14)*(m1)))))), svex2_40);
        svex2_41.v0 = svsub_f64_x(pg1, t1740, t1745);
        svex2_41.v1 = svadd_f64_x(pg1, t1741, t1744);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((15)*(m1)))))), svex2_41);
        svex2_42.v0 = svsub_f64_x(pg1, t1764, t1769);
        svex2_42.v1 = svadd_f64_x(pg1, t1765, t1768);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((16)*(m1)))))), svex2_42);
        svex2_43.v0 = svsub_f64_x(pg1, t1788, t1793);
        svex2_43.v1 = svadd_f64_x(pg1, t1789, t1792);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((17)*(m1)))))), svex2_43);
        svex2_44.v0 = svsub_f64_x(pg1, t1812, t1817);
        svex2_44.v1 = svadd_f64_x(pg1, t1813, t1816);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((18)*(m1)))))), svex2_44);
        svex2_45.v0 = svsub_f64_x(pg1, t1836, t1841);
        svex2_45.v1 = svadd_f64_x(pg1, t1837, t1840);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((19)*(m1)))))), svex2_45);
        svex2_46.v0 = svsub_f64_x(pg1, t1738, t1742);
        svex2_46.v1 = svsub_f64_x(pg1, t1739, t1743);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((20)*(m1)))))), svex2_46);
        svex2_47.v0 = svsub_f64_x(pg1, t1762, t1766);
        svex2_47.v1 = svsub_f64_x(pg1, t1763, t1767);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((21)*(m1)))))), svex2_47);
        svex2_48.v0 = svsub_f64_x(pg1, t1786, t1790);
        svex2_48.v1 = svsub_f64_x(pg1, t1787, t1791);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((22)*(m1)))))), svex2_48);
        svex2_49.v0 = svsub_f64_x(pg1, t1810, t1814);
        svex2_49.v1 = svsub_f64_x(pg1, t1811, t1815);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((23)*(m1)))))), svex2_49);
        svex2_50.v0 = svsub_f64_x(pg1, t1834, t1838);
        svex2_50.v1 = svsub_f64_x(pg1, t1835, t1839);
        svst2_f64(pg1, (Y + ((2)*((k1 + ((24)*(m1)))))), svex2_50);
        k1 += svcntd();
        pg1 = svwhilelt_b64(k1, m1);
    } while(svptest_any(svptrue_b64(), pg1));
}