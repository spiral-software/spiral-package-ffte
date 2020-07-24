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


void dft8b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    double a870, a871, a872, a873, a875, a876, a877, a878, 
            a879, a880, a881, a882, a883, a884, a885, a886, 
            a887, a888, s183, s184, s185, s186, s187, s188, 
            s189, s190, s191, s192, s193, s194, s195, s196, 
            s197, s198, s199, s200, s201, s202, s203, s204, 
            s205, s206, s207, s208, s209, s210, s211, s212, 
            s213, s214, s215, s216, t614, t615, t616, t617, 
            t618, t619, t620, t621, t622, t623, t624, t625, 
            t626, t627, t628, t629, t630, t631, t632, t633, 
            t634, t635, t636, t637, t638, t639, t640, t641;
    int a860, a861, a862, a863, a864, a865, a866, a867, 
            a868, a869, a874, a889, a890, a891, a892, a893, 
            a894, a895, a896, a897, b120, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a860 = (2*k1);
            a861 = (j1*m1);
            a862 = (a860 + (2*a861));
            s183 = X[a862];
            s184 = X[(a862 + 1)];
            b120 = (l1*m1);
            a863 = (a862 + (2*b120));
            s185 = X[a863];
            s186 = X[(a863 + 1)];
            a864 = (a862 + (4*b120));
            s187 = X[a864];
            s188 = X[(a864 + 1)];
            a865 = (a862 + (6*b120));
            s189 = X[a865];
            s190 = X[(a865 + 1)];
            a866 = (a862 + (8*b120));
            s191 = X[a866];
            s192 = X[(a866 + 1)];
            a867 = (a862 + (10*b120));
            s193 = X[a867];
            s194 = X[(a867 + 1)];
            a868 = (a862 + (12*b120));
            s195 = X[a868];
            s196 = X[(a868 + 1)];
            a869 = (a862 + (14*b120));
            s197 = X[a869];
            s198 = X[(a869 + 1)];
            t614 = (s183 + s191);
            t615 = (s184 + s192);
            t616 = (s183 - s191);
            t617 = (s184 - s192);
            t618 = (s187 + s195);
            t619 = (s188 + s196);
            t620 = (s187 - s195);
            t621 = (s188 - s196);
            t622 = (t614 + t618);
            t623 = (t615 + t619);
            t624 = (t614 - t618);
            t625 = (t615 - t619);
            t626 = (t616 + t621);
            t627 = (t617 - t620);
            t628 = (t616 - t621);
            t629 = (t617 + t620);
            t630 = (s185 + s193);
            t631 = (s186 + s194);
            t632 = (s185 - s193);
            t633 = (s186 - s194);
            t634 = (s189 + s197);
            t635 = (s190 + s198);
            t636 = (s189 - s197);
            t637 = (s190 - s198);
            t638 = (t630 + t634);
            t639 = (t631 + t635);
            t640 = (t630 - t634);
            t641 = (t631 - t635);
            a870 = (0.70710678118654757*(t632 + t637));
            a871 = (0.70710678118654757*(t633 - t636));
            s199 = (a870 + a871);
            s200 = (a871 - a870);
            a872 = (0.70710678118654757*(t633 + t636));
            a873 = (0.70710678118654757*(t632 - t637));
            s201 = (a872 - a873);
            s202 = (a873 + a872);
            s203 = (t622 - t638);
            s204 = (t623 - t639);
            s205 = (t626 + s199);
            s206 = (t627 + s200);
            s207 = (t626 - s199);
            s208 = (t627 - s200);
            s209 = (t624 + t641);
            s210 = (t625 - t640);
            s211 = (t624 - t641);
            s212 = (t625 + t640);
            s213 = (t628 + s201);
            s214 = (t629 - s202);
            s215 = (t628 - s201);
            s216 = (t629 + s202);
            a874 = (14*j1);
            a875 = TW1[a874];
            a876 = TW1[(a874 + 1)];
            a877 = TW1[(a874 + 2)];
            a878 = TW1[(a874 + 3)];
            a879 = TW1[(a874 + 4)];
            a880 = TW1[(a874 + 5)];
            a881 = TW1[(a874 + 6)];
            a882 = TW1[(a874 + 7)];
            a883 = TW1[(a874 + 8)];
            a884 = TW1[(a874 + 9)];
            a885 = TW1[(a874 + 10)];
            a886 = TW1[(a874 + 11)];
            a887 = TW1[(a874 + 12)];
            a888 = TW1[(a874 + 13)];
            a889 = (16*a861);
            a890 = (a860 + a889);
            Y[a890] = (t622 + t638);
            Y[(a890 + 1)] = (t623 + t639);
            a891 = (a860 + (2*m1) + a889);
            Y[a891] = ((a875*s205) - (a876*s206));
            Y[(a891 + 1)] = ((a876*s205) + (a875*s206));
            a892 = (a860 + (4*m1) + a889);
            Y[a892] = ((a877*s209) - (a878*s210));
            Y[(a892 + 1)] = ((a878*s209) + (a877*s210));
            a893 = (a860 + (6*m1) + a889);
            Y[a893] = ((a879*s213) - (a880*s214));
            Y[(a893 + 1)] = ((a880*s213) + (a879*s214));
            a894 = (a860 + (8*m1) + a889);
            Y[a894] = ((a881*s203) - (a882*s204));
            Y[(a894 + 1)] = ((a882*s203) + (a881*s204));
            a895 = (a860 + (10*m1) + a889);
            Y[a895] = ((a883*s207) - (a884*s208));
            Y[(a895 + 1)] = ((a884*s207) + (a883*s208));
            a896 = (a860 + (12*m1) + a889);
            Y[a896] = ((a885*s211) - (a886*s212));
            Y[(a896 + 1)] = ((a886*s211) + (a885*s212));
            a897 = (a860 + (14*m1) + a889);
            Y[a897] = ((a887*s215) - (a888*s216));
            Y[(a897 + 1)] = ((a888*s215) + (a887*s216));
        }
    }
}