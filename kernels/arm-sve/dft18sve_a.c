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

void dft18a_(float64_t  *Y, float64_t  *X, float64_t  *TW1, int  *lp1) {
    static float64_t D39[12];
    static float64_t D40[24];
    int l1;
    float64_t  *a3518;
    svfloat64x2_t s1758, s1761, s1764, s1767, s1770, s1773, s1776, s1779, 
            s1782, s1785, s1788, s1791, s1794, s1797, s1800, s1803, 
            s1806, s1809;
    svfloat64_t a3519, a3520, a3521, a3522, a3523, a3524, a3525, a3526, 
            a3527, a3528, a3529, a3530, a3531, a3532, a3533, a3534, 
            a3535, a3536, a3537, a3538, a3539, a3540, a3541, a3542, 
            a3543, a3544, a3545, a3546, a3547, a3548, a3549, a3550, 
            a3551, a3552, s1759, s1760, s1762, s1763, s1765, s1766, 
            s1768, s1769, s1771, s1772, s1774, s1775, s1777, s1778, 
            s1780, s1781, s1783, s1784, s1786, s1787, s1789, s1790, 
            s1792, s1793, s1795, s1796, s1798, s1799, s1801, s1802, 
            s1804, s1805, s1807, s1808, s1810, s1811, s1812, s1813, 
            s1814, s1815, s1816, s1817, s1818, s1819, s1820, s1821, 
            s1822, s1823, s1824, s1825, s1826, s1827, s1828, s1829, 
            s1830, s1831, s1832, s1833, s1834, s1835, s1836, s1837, 
            s1838, s1839, s1840, s1841, s1842, s1843, s1844, s1845, 
            s1846, s1847, s1848, s1849, s1850, s1851, s1852, s1853, 
            s1854, s1855, s1856, s1857, s1858, s1859, s1860, s1861, 
            s1862, s1863, s1864, s1865, s1866, s1867, s1868, s1869, 
            s1870, s1871, s1872, s1873, s1874, s1875, s1876, s1877, 
            s1878, s1879, s1880, s1881, s1882, s1883, s1884, s1885, 
            s1886, s1887, s1888, s1889, s1890, s1891, s1892, s1893, 
            s1894, s1895, s1896, s1897, s1898, s1899, s1900, s1901, 
            s1902, s1903, s1904, s1905, s1906, s1907, s1908, s1909, 
            s1910, s1911, s1912, s1913, s1914, s1915, s1916, s1917, 
            s1918, s1919, s1920, s1921, s1922, s1923, s1924, s1925, 
            s1926, s1927, s1928, s1929, s1930, s1931, s1932, s1933, 
            s1934, s1935, s1936, s1937, s1938, s1939, s1940, s1941, 
            s1942, s1943, s1944, s1945, s1946, s1947, s1948, s1949, 
            s1950, s1951, s1952, s1953, s1954, s1955, s1956, s1957, 
            s1958, s1959, s1960, s1961, s1962, s1963, s1964, s1965, 
            t2075, t2076, t2077, t2078, t2079, t2080, t2081, t2082, 
            t2083, t2084, t2085, t2086, t2087, t2088, t2089, t2090, 
            t2091, t2092, t2093, t2094, t2095, t2096, t2097, t2098, 
            t2099, t2100, t2101, t2102, t2103, t2104, t2105, t2106, 
            t2107, t2108, t2109, t2110, t2111, t2112, t2113, t2114, 
            t2115, t2116, t2117, t2118, t2119, t2120, t2121, t2122, 
            t2123, t2124, t2125, t2126, t2127, t2128, t2129, t2130, 
            t2131, t2132, t2133, t2134, t2135, t2136, t2137, t2138, 
            t2139, t2140, t2141, t2142, t2143, t2144, t2145, t2146, 
            t2147, t2148, t2149, t2150, t2151, t2152, t2153, t2154, 
            t2155, t2156, t2157, t2158, t2159, t2160, t2161, t2162, 
            t2163, t2164, t2165, t2166, t2167, t2168, t2169, t2170, 
            t2171, t2172, t2173, t2174, t2175, t2176, t2177, t2178, 
            t2179, t2180, t2181, t2182, t2183, t2184, t2185, t2186, 
            t2187, t2188, t2189, t2190, t2191, t2192, t2193, t2194;
    svbool_t pg1;
    l1 = *(lp1);
    int j1 = 0;
    pg1 = svwhilelt_b64(j1, l1);
    do {
        s1758 = svld2_f64(pg1, (X + ((2)*(j1))));
        s1759 = s1758.v0;
        s1760 = s1758.v1;
        s1761 = svld2_f64(pg1, (X + ((2)*((j1 + l1)))));
        s1762 = s1761.v0;
        s1763 = s1761.v1;
        s1764 = svld2_f64(pg1, (X + ((2)*((j1 + ((2)*(l1)))))));
        s1765 = s1764.v0;
        s1766 = s1764.v1;
        s1767 = svld2_f64(pg1, (X + ((2)*((j1 + ((3)*(l1)))))));
        s1768 = s1767.v0;
        s1769 = s1767.v1;
        s1770 = svld2_f64(pg1, (X + ((2)*((j1 + ((4)*(l1)))))));
        s1771 = s1770.v0;
        s1772 = s1770.v1;
        s1773 = svld2_f64(pg1, (X + ((2)*((j1 + ((5)*(l1)))))));
        s1774 = s1773.v0;
        s1775 = s1773.v1;
        s1776 = svld2_f64(pg1, (X + ((2)*((j1 + ((6)*(l1)))))));
        s1777 = s1776.v0;
        s1778 = s1776.v1;
        s1779 = svld2_f64(pg1, (X + ((2)*((j1 + ((7)*(l1)))))));
        s1780 = s1779.v0;
        s1781 = s1779.v1;
        s1782 = svld2_f64(pg1, (X + ((2)*((j1 + ((8)*(l1)))))));
        s1783 = s1782.v0;
        s1784 = s1782.v1;
        s1785 = svld2_f64(pg1, (X + ((2)*((j1 + ((9)*(l1)))))));
        s1786 = s1785.v0;
        s1787 = s1785.v1;
        s1788 = svld2_f64(pg1, (X + ((2)*((j1 + ((10)*(l1)))))));
        s1789 = s1788.v0;
        s1790 = s1788.v1;
        s1791 = svld2_f64(pg1, (X + ((2)*((j1 + ((11)*(l1)))))));
        s1792 = s1791.v0;
        s1793 = s1791.v1;
        s1794 = svld2_f64(pg1, (X + ((2)*((j1 + ((12)*(l1)))))));
        s1795 = s1794.v0;
        s1796 = s1794.v1;
        s1797 = svld2_f64(pg1, (X + ((2)*((j1 + ((13)*(l1)))))));
        s1798 = s1797.v0;
        s1799 = s1797.v1;
        s1800 = svld2_f64(pg1, (X + ((2)*((j1 + ((14)*(l1)))))));
        s1801 = s1800.v0;
        s1802 = s1800.v1;
        s1803 = svld2_f64(pg1, (X + ((2)*((j1 + ((15)*(l1)))))));
        s1804 = s1803.v0;
        s1805 = s1803.v1;
        s1806 = svld2_f64(pg1, (X + ((2)*((j1 + ((16)*(l1)))))));
        s1807 = s1806.v0;
        s1808 = s1806.v1;
        s1809 = svld2_f64(pg1, (X + ((2)*((j1 + ((17)*(l1)))))));
        s1810 = s1809.v0;
        s1811 = s1809.v1;
        t2075 = svadd_f64_x(pg1, s1777, s1795);
        t2076 = svadd_f64_x(pg1, s1778, s1796);
        t2077 = svadd_f64_x(pg1, s1759, t2075);
        t2078 = svadd_f64_x(pg1, s1760, t2076);
        t2079 = svmls_n_f64_x(pg1, s1759, t2075, 0.5);
        t2080 = svmls_n_f64_x(pg1, s1760, t2076, 0.5);
        s1812 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1778, s1796), 0.8660254037844386);
        s1813 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1777, s1795), 0.8660254037844386);
        s1814 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2078, D39[1]), t2077, D39[0]);
        s1815 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2078, D39[0]), t2077, D39[1]);
        t2081 = svadd_f64_x(pg1, t2079, s1812);
        t2082 = svsub_f64_x(pg1, t2080, s1813);
        t2083 = svsub_f64_x(pg1, t2079, s1812);
        t2084 = svadd_f64_x(pg1, t2080, s1813);
        s1816 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2082, D40[1]), t2081, D40[0]);
        s1817 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2082, D40[0]), t2081, D40[1]);
        s1818 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2084, D40[3]), t2083, D40[2]);
        s1819 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2084, D40[2]), t2083, D40[3]);
        t2085 = svadd_f64_x(pg1, s1780, s1798);
        t2086 = svadd_f64_x(pg1, s1781, s1799);
        t2087 = svadd_f64_x(pg1, s1762, t2085);
        t2088 = svadd_f64_x(pg1, s1763, t2086);
        t2089 = svmls_n_f64_x(pg1, s1762, t2085, 0.5);
        t2090 = svmls_n_f64_x(pg1, s1763, t2086, 0.5);
        s1820 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1781, s1799), 0.8660254037844386);
        s1821 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1780, s1798), 0.8660254037844386);
        s1822 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2088, D39[3]), t2087, D39[2]);
        s1823 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2088, D39[2]), t2087, D39[3]);
        t2091 = svadd_f64_x(pg1, t2089, s1820);
        t2092 = svsub_f64_x(pg1, t2090, s1821);
        t2093 = svsub_f64_x(pg1, t2089, s1820);
        t2094 = svadd_f64_x(pg1, t2090, s1821);
        s1824 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2092, D40[5]), t2091, D40[4]);
        s1825 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2092, D40[4]), t2091, D40[5]);
        s1826 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2094, D40[7]), t2093, D40[6]);
        s1827 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2094, D40[6]), t2093, D40[7]);
        t2095 = svadd_f64_x(pg1, s1783, s1801);
        t2096 = svadd_f64_x(pg1, s1784, s1802);
        t2097 = svadd_f64_x(pg1, s1765, t2095);
        t2098 = svadd_f64_x(pg1, s1766, t2096);
        t2099 = svmls_n_f64_x(pg1, s1765, t2095, 0.5);
        t2100 = svmls_n_f64_x(pg1, s1766, t2096, 0.5);
        s1828 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1784, s1802), 0.8660254037844386);
        s1829 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1783, s1801), 0.8660254037844386);
        s1830 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2098, D39[5]), t2097, D39[4]);
        s1831 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2098, D39[4]), t2097, D39[5]);
        t2101 = svadd_f64_x(pg1, t2099, s1828);
        t2102 = svsub_f64_x(pg1, t2100, s1829);
        t2103 = svsub_f64_x(pg1, t2099, s1828);
        t2104 = svadd_f64_x(pg1, t2100, s1829);
        s1832 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2102, D40[9]), t2101, D40[8]);
        s1833 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2102, D40[8]), t2101, D40[9]);
        s1834 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2104, D40[11]), t2103, D40[10]);
        s1835 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2104, D40[10]), t2103, D40[11]);
        t2105 = svadd_f64_x(pg1, s1786, s1804);
        t2106 = svadd_f64_x(pg1, s1787, s1805);
        t2107 = svadd_f64_x(pg1, s1768, t2105);
        t2108 = svadd_f64_x(pg1, s1769, t2106);
        t2109 = svmls_n_f64_x(pg1, s1768, t2105, 0.5);
        t2110 = svmls_n_f64_x(pg1, s1769, t2106, 0.5);
        s1836 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1787, s1805), 0.8660254037844386);
        s1837 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1786, s1804), 0.8660254037844386);
        s1838 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2108, D39[7]), t2107, D39[6]);
        s1839 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2108, D39[6]), t2107, D39[7]);
        t2111 = svadd_f64_x(pg1, t2109, s1836);
        t2112 = svsub_f64_x(pg1, t2110, s1837);
        t2113 = svsub_f64_x(pg1, t2109, s1836);
        t2114 = svadd_f64_x(pg1, t2110, s1837);
        s1840 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2112, D40[13]), t2111, D40[12]);
        s1841 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2112, D40[12]), t2111, D40[13]);
        s1842 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2114, D40[15]), t2113, D40[14]);
        s1843 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2114, D40[14]), t2113, D40[15]);
        t2115 = svadd_f64_x(pg1, s1789, s1807);
        t2116 = svadd_f64_x(pg1, s1790, s1808);
        t2117 = svadd_f64_x(pg1, s1771, t2115);
        t2118 = svadd_f64_x(pg1, s1772, t2116);
        t2119 = svmls_n_f64_x(pg1, s1771, t2115, 0.5);
        t2120 = svmls_n_f64_x(pg1, s1772, t2116, 0.5);
        s1844 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1790, s1808), 0.8660254037844386);
        s1845 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1789, s1807), 0.8660254037844386);
        s1846 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2118, D39[9]), t2117, D39[8]);
        s1847 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2118, D39[8]), t2117, D39[9]);
        t2121 = svadd_f64_x(pg1, t2119, s1844);
        t2122 = svsub_f64_x(pg1, t2120, s1845);
        t2123 = svsub_f64_x(pg1, t2119, s1844);
        t2124 = svadd_f64_x(pg1, t2120, s1845);
        s1848 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2122, D40[17]), t2121, D40[16]);
        s1849 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2122, D40[16]), t2121, D40[17]);
        s1850 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2124, D40[19]), t2123, D40[18]);
        s1851 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2124, D40[18]), t2123, D40[19]);
        t2125 = svadd_f64_x(pg1, s1792, s1810);
        t2126 = svadd_f64_x(pg1, s1793, s1811);
        t2127 = svadd_f64_x(pg1, s1774, t2125);
        t2128 = svadd_f64_x(pg1, s1775, t2126);
        t2129 = svmls_n_f64_x(pg1, s1774, t2125, 0.5);
        t2130 = svmls_n_f64_x(pg1, s1775, t2126, 0.5);
        s1852 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1793, s1811), 0.8660254037844386);
        s1853 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1792, s1810), 0.8660254037844386);
        s1854 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2128, D39[11]), t2127, D39[10]);
        s1855 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2128, D39[10]), t2127, D39[11]);
        t2131 = svadd_f64_x(pg1, t2129, s1852);
        t2132 = svsub_f64_x(pg1, t2130, s1853);
        t2133 = svsub_f64_x(pg1, t2129, s1852);
        t2134 = svadd_f64_x(pg1, t2130, s1853);
        s1856 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2132, D40[21]), t2131, D40[20]);
        s1857 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2132, D40[20]), t2131, D40[21]);
        s1858 = svnmls_n_f64_x(pg1, svmul_n_f64_x(pg1, t2134, D40[23]), t2133, D40[22]);
        s1859 = svmla_n_f64_x(pg1, svmul_n_f64_x(pg1, t2134, D40[22]), t2133, D40[23]);
        t2135 = svadd_f64_x(pg1, s1830, s1846);
        t2136 = svadd_f64_x(pg1, s1831, s1847);
        t2137 = svadd_f64_x(pg1, s1814, t2135);
        t2138 = svadd_f64_x(pg1, s1815, t2136);
        t2139 = svmls_n_f64_x(pg1, s1814, t2135, 0.5);
        t2140 = svmls_n_f64_x(pg1, s1815, t2136, 0.5);
        s1860 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1831, s1847), 0.8660254037844386);
        s1861 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1830, s1846), 0.8660254037844386);
        t2141 = svadd_f64_x(pg1, t2139, s1860);
        t2142 = svsub_f64_x(pg1, t2140, s1861);
        t2143 = svsub_f64_x(pg1, t2139, s1860);
        t2144 = svadd_f64_x(pg1, t2140, s1861);
        t2145 = svadd_f64_x(pg1, s1838, s1854);
        t2146 = svadd_f64_x(pg1, s1839, s1855);
        t2147 = svadd_f64_x(pg1, s1822, t2145);
        t2148 = svadd_f64_x(pg1, s1823, t2146);
        t2149 = svmls_n_f64_x(pg1, s1822, t2145, 0.5);
        t2150 = svmls_n_f64_x(pg1, s1823, t2146, 0.5);
        s1862 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1839, s1855), 0.8660254037844386);
        s1863 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1838, s1854), 0.8660254037844386);
        t2151 = svadd_f64_x(pg1, t2149, s1862);
        t2152 = svsub_f64_x(pg1, t2150, s1863);
        t2153 = svsub_f64_x(pg1, t2149, s1862);
        t2154 = svadd_f64_x(pg1, t2150, s1863);
        s1954 = svmla_n_f64_x(pg1, t2151, t2152, 1.7320508075688772);
        s1864 = svmul_n_f64_x(pg1, s1954, 0.5);
        s1955 = svmls_n_f64_x(pg1, t2152, t2151, 1.7320508075688772);
        s1865 = svmul_n_f64_x(pg1, s1955, 0.5);
        s1956 = svmls_n_f64_x(pg1, t2154, t2153, 0.57735026918962584);
        s1866 = svmul_n_f64_x(pg1, s1956, 0.8660254037844386);
        s1957 = svmla_n_f64_x(pg1, t2153, t2154, 0.57735026918962584);
        s1867 = svmul_n_f64_x(pg1, s1957, 0.8660254037844386);
        s1868 = svadd_f64_x(pg1, t2137, t2147);
        s1869 = svadd_f64_x(pg1, t2138, t2148);
        s1870 = svsub_f64_x(pg1, t2137, t2147);
        s1871 = svsub_f64_x(pg1, t2138, t2148);
        s1872 = svadd_f64_x(pg1, t2141, s1864);
        s1873 = svadd_f64_x(pg1, t2142, s1865);
        s1874 = svsub_f64_x(pg1, t2141, s1864);
        s1875 = svsub_f64_x(pg1, t2142, s1865);
        s1876 = svadd_f64_x(pg1, t2143, s1866);
        s1877 = svsub_f64_x(pg1, t2144, s1867);
        s1878 = svsub_f64_x(pg1, t2143, s1866);
        s1879 = svadd_f64_x(pg1, t2144, s1867);
        t2155 = svadd_f64_x(pg1, s1832, s1848);
        t2156 = svadd_f64_x(pg1, s1833, s1849);
        t2157 = svadd_f64_x(pg1, s1816, t2155);
        t2158 = svadd_f64_x(pg1, s1817, t2156);
        t2159 = svmls_n_f64_x(pg1, s1816, t2155, 0.5);
        t2160 = svmls_n_f64_x(pg1, s1817, t2156, 0.5);
        s1880 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1833, s1849), 0.8660254037844386);
        s1881 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1832, s1848), 0.8660254037844386);
        t2161 = svadd_f64_x(pg1, t2159, s1880);
        t2162 = svsub_f64_x(pg1, t2160, s1881);
        t2163 = svsub_f64_x(pg1, t2159, s1880);
        t2164 = svadd_f64_x(pg1, t2160, s1881);
        t2165 = svadd_f64_x(pg1, s1840, s1856);
        t2166 = svadd_f64_x(pg1, s1841, s1857);
        t2167 = svadd_f64_x(pg1, s1824, t2165);
        t2168 = svadd_f64_x(pg1, s1825, t2166);
        t2169 = svmls_n_f64_x(pg1, s1824, t2165, 0.5);
        t2170 = svmls_n_f64_x(pg1, s1825, t2166, 0.5);
        s1882 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1841, s1857), 0.8660254037844386);
        s1883 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1840, s1856), 0.8660254037844386);
        t2171 = svadd_f64_x(pg1, t2169, s1882);
        t2172 = svsub_f64_x(pg1, t2170, s1883);
        t2173 = svsub_f64_x(pg1, t2169, s1882);
        t2174 = svadd_f64_x(pg1, t2170, s1883);
        s1958 = svmla_n_f64_x(pg1, t2171, t2172, 1.7320508075688772);
        s1884 = svmul_n_f64_x(pg1, s1958, 0.5);
        s1959 = svmls_n_f64_x(pg1, t2172, t2171, 1.7320508075688772);
        s1885 = svmul_n_f64_x(pg1, s1959, 0.5);
        s1960 = svmls_n_f64_x(pg1, t2174, t2173, 0.57735026918962584);
        s1886 = svmul_n_f64_x(pg1, s1960, 0.8660254037844386);
        s1961 = svmla_n_f64_x(pg1, t2173, t2174, 0.57735026918962584);
        s1887 = svmul_n_f64_x(pg1, s1961, 0.8660254037844386);
        s1888 = svadd_f64_x(pg1, t2157, t2167);
        s1889 = svadd_f64_x(pg1, t2158, t2168);
        s1890 = svsub_f64_x(pg1, t2157, t2167);
        s1891 = svsub_f64_x(pg1, t2158, t2168);
        s1892 = svadd_f64_x(pg1, t2161, s1884);
        s1893 = svadd_f64_x(pg1, t2162, s1885);
        s1894 = svsub_f64_x(pg1, t2161, s1884);
        s1895 = svsub_f64_x(pg1, t2162, s1885);
        s1896 = svadd_f64_x(pg1, t2163, s1886);
        s1897 = svsub_f64_x(pg1, t2164, s1887);
        s1898 = svsub_f64_x(pg1, t2163, s1886);
        s1899 = svadd_f64_x(pg1, t2164, s1887);
        t2175 = svadd_f64_x(pg1, s1834, s1850);
        t2176 = svadd_f64_x(pg1, s1835, s1851);
        t2177 = svadd_f64_x(pg1, s1818, t2175);
        t2178 = svadd_f64_x(pg1, s1819, t2176);
        t2179 = svmls_n_f64_x(pg1, s1818, t2175, 0.5);
        t2180 = svmls_n_f64_x(pg1, s1819, t2176, 0.5);
        s1900 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1835, s1851), 0.8660254037844386);
        s1901 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1834, s1850), 0.8660254037844386);
        t2181 = svadd_f64_x(pg1, t2179, s1900);
        t2182 = svsub_f64_x(pg1, t2180, s1901);
        t2183 = svsub_f64_x(pg1, t2179, s1900);
        t2184 = svadd_f64_x(pg1, t2180, s1901);
        t2185 = svadd_f64_x(pg1, s1842, s1858);
        t2186 = svadd_f64_x(pg1, s1843, s1859);
        t2187 = svadd_f64_x(pg1, s1826, t2185);
        t2188 = svadd_f64_x(pg1, s1827, t2186);
        t2189 = svmls_n_f64_x(pg1, s1826, t2185, 0.5);
        t2190 = svmls_n_f64_x(pg1, s1827, t2186, 0.5);
        s1902 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1843, s1859), 0.8660254037844386);
        s1903 = svmul_n_f64_x(pg1, svsub_f64_x(pg1, s1842, s1858), 0.8660254037844386);
        t2191 = svadd_f64_x(pg1, t2189, s1902);
        t2192 = svsub_f64_x(pg1, t2190, s1903);
        t2193 = svsub_f64_x(pg1, t2189, s1902);
        t2194 = svadd_f64_x(pg1, t2190, s1903);
        s1962 = svmla_n_f64_x(pg1, t2191, t2192, 1.7320508075688772);
        s1904 = svmul_n_f64_x(pg1, s1962, 0.5);
        s1963 = svmls_n_f64_x(pg1, t2192, t2191, 1.7320508075688772);
        s1905 = svmul_n_f64_x(pg1, s1963, 0.5);
        s1964 = svmls_n_f64_x(pg1, t2194, t2193, 0.57735026918962584);
        s1906 = svmul_n_f64_x(pg1, s1964, 0.8660254037844386);
        s1965 = svmla_n_f64_x(pg1, t2193, t2194, 0.57735026918962584);
        s1907 = svmul_n_f64_x(pg1, s1965, 0.8660254037844386);
        s1908 = svadd_f64_x(pg1, t2177, t2187);
        s1909 = svadd_f64_x(pg1, t2178, t2188);
        s1910 = svsub_f64_x(pg1, t2177, t2187);
        s1911 = svsub_f64_x(pg1, t2178, t2188);
        s1912 = svadd_f64_x(pg1, t2181, s1904);
        s1913 = svadd_f64_x(pg1, t2182, s1905);
        s1914 = svsub_f64_x(pg1, t2181, s1904);
        s1915 = svsub_f64_x(pg1, t2182, s1905);
        s1916 = svadd_f64_x(pg1, t2183, s1906);
        s1917 = svsub_f64_x(pg1, t2184, s1907);
        s1918 = svsub_f64_x(pg1, t2183, s1906);
        s1919 = svadd_f64_x(pg1, t2184, s1907);
        a3518 = (TW1 + ((34)*(j1)));
        a3519 = svld1_gather_u64offset_f64(pg1, a3518, svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3520 = svld1_gather_u64offset_f64(pg1, (a3518 + 1), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1920 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3520, s1889), a3519, s1888);
        s1921 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3519, s1889), a3520, s1888);
        a3521 = svld1_gather_u64offset_f64(pg1, (a3518 + 2), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3522 = svld1_gather_u64offset_f64(pg1, (a3518 + 3), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1922 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3522, s1909), a3521, s1908);
        s1923 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3521, s1909), a3522, s1908);
        a3523 = svld1_gather_u64offset_f64(pg1, (a3518 + 4), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3524 = svld1_gather_u64offset_f64(pg1, (a3518 + 5), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1924 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3524, s1873), a3523, s1872);
        s1925 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3523, s1873), a3524, s1872);
        a3525 = svld1_gather_u64offset_f64(pg1, (a3518 + 6), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3526 = svld1_gather_u64offset_f64(pg1, (a3518 + 7), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1926 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3526, s1893), a3525, s1892);
        s1927 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3525, s1893), a3526, s1892);
        a3527 = svld1_gather_u64offset_f64(pg1, (a3518 + 8), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3528 = svld1_gather_u64offset_f64(pg1, (a3518 + 9), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1928 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3528, s1913), a3527, s1912);
        s1929 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3527, s1913), a3528, s1912);
        a3529 = svld1_gather_u64offset_f64(pg1, (a3518 + 10), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3530 = svld1_gather_u64offset_f64(pg1, (a3518 + 11), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1930 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3530, s1877), a3529, s1876);
        s1931 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3529, s1877), a3530, s1876);
        a3531 = svld1_gather_u64offset_f64(pg1, (a3518 + 12), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3532 = svld1_gather_u64offset_f64(pg1, (a3518 + 13), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1932 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3532, s1897), a3531, s1896);
        s1933 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3531, s1897), a3532, s1896);
        a3533 = svld1_gather_u64offset_f64(pg1, (a3518 + 14), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3534 = svld1_gather_u64offset_f64(pg1, (a3518 + 15), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1934 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3534, s1917), a3533, s1916);
        s1935 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3533, s1917), a3534, s1916);
        a3535 = svld1_gather_u64offset_f64(pg1, (a3518 + 16), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3536 = svld1_gather_u64offset_f64(pg1, (a3518 + 17), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1936 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3536, s1871), a3535, s1870);
        s1937 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3535, s1871), a3536, s1870);
        a3537 = svld1_gather_u64offset_f64(pg1, (a3518 + 18), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3538 = svld1_gather_u64offset_f64(pg1, (a3518 + 19), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1938 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3538, s1891), a3537, s1890);
        s1939 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3537, s1891), a3538, s1890);
        a3539 = svld1_gather_u64offset_f64(pg1, (a3518 + 20), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3540 = svld1_gather_u64offset_f64(pg1, (a3518 + 21), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1940 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3540, s1911), a3539, s1910);
        s1941 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3539, s1911), a3540, s1910);
        a3541 = svld1_gather_u64offset_f64(pg1, (a3518 + 22), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3542 = svld1_gather_u64offset_f64(pg1, (a3518 + 23), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1942 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3542, s1875), a3541, s1874);
        s1943 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3541, s1875), a3542, s1874);
        a3543 = svld1_gather_u64offset_f64(pg1, (a3518 + 24), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3544 = svld1_gather_u64offset_f64(pg1, (a3518 + 25), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1944 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3544, s1895), a3543, s1894);
        s1945 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3543, s1895), a3544, s1894);
        a3545 = svld1_gather_u64offset_f64(pg1, (a3518 + 26), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3546 = svld1_gather_u64offset_f64(pg1, (a3518 + 27), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1946 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3546, s1915), a3545, s1914);
        s1947 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3545, s1915), a3546, s1914);
        a3547 = svld1_gather_u64offset_f64(pg1, (a3518 + 28), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3548 = svld1_gather_u64offset_f64(pg1, (a3518 + 29), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1948 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3548, s1879), a3547, s1878);
        s1949 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3547, s1879), a3548, s1878);
        a3549 = svld1_gather_u64offset_f64(pg1, (a3518 + 30), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3550 = svld1_gather_u64offset_f64(pg1, (a3518 + 31), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1950 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3550, s1899), a3549, s1898);
        s1951 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3549, s1899), a3550, s1898);
        a3551 = svld1_gather_u64offset_f64(pg1, (a3518 + 32), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        a3552 = svld1_gather_u64offset_f64(pg1, (a3518 + 33), svindex_u64(0, (int64_t)(34 * sizeof(float64_t))));
        s1952 = svnmls_f64_x(pg1, svmul_f64_x(pg1, a3552, s1919), a3551, s1918);
        s1953 = svmla_f64_x(pg1, svmul_f64_x(pg1, a3551, s1919), a3552, s1918);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1868);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(1 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1869);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(2 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1920);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(3 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1921);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(4 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1922);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(5 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1923);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(6 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1924);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(7 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1925);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(8 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1926);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(9 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1927);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(10 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1928);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(11 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1929);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(12 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1930);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(13 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1931);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(14 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1932);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(15 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1933);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(16 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1934);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(17 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1935);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(18 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1936);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(19 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1937);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(20 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1938);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(21 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1939);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(22 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1940);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(23 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1941);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(24 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1942);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(25 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1943);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(26 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1944);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(27 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1945);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(28 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1946);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(29 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1947);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(30 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1948);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(31 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1949);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(32 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1950);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(33 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1951);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(34 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1952);
        svst1_scatter_u64offset_f64(pg1, ((float64_t  *)(35 + Y + ((36)*(j1)))), svindex_u64(0, (int64_t)(36 * sizeof(float64_t))), s1953);
        j1 += svcntd();
        pg1 = svwhilelt_b64(j1, l1);
    } while(svptest_any(svptrue_b64(), pg1));
}
