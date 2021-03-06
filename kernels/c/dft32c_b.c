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


void dft32b_(double  *Y, double  *X, double  *TW1, int  *lp1, int  *mp1) {
    static double D15[64];
    double a10926, a10927, a10928, a10929, a10930, a10931, a10932, a10933, 
            a10934, a10935, a10936, a10937, a10938, a10939, a10940, a10941, 
            a10943, a10944, a10945, a10946, a10947, a10948, a10949, a10950, 
            a10951, a10952, a10953, a10954, a10955, a10956, a10957, a10958, 
            a10959, a10960, a10961, a10962, a10963, a10964, a10965, a10966, 
            a10967, a10968, a10969, a10970, a10971, a10972, a10973, a10974, 
            a10975, a10976, a10977, a10978, a10979, a10980, a10981, a10982, 
            a10983, a10984, a10985, a10986, a10987, a10988, a10989, a10990, 
            a10991, a10992, a10993, a10994, a10995, a10996, a10997, a10998, 
            a10999, a11000, a11001, a11002, a11003, a11004, s3988, s3989, 
            s3990, s3991, s3992, s3993, s3994, s3995, s3996, s3997, 
            s3998, s3999, s4000, s4001, s4002, s4003, s4004, s4005, 
            s4006, s4007, s4008, s4009, s4010, s4011, s4012, s4013, 
            s4014, s4015, s4016, s4017, s4018, s4019, s4020, s4021, 
            s4022, s4023, s4024, s4025, s4026, s4027, s4028, s4029, 
            s4030, s4031, s4032, s4033, s4034, s4035, s4036, s4037, 
            s4038, s4039, s4040, s4041, s4042, s4043, s4044, s4045, 
            s4046, s4047, s4048, s4049, s4050, s4051, s4052, s4053, 
            s4054, s4055, s4056, s4057, s4058, s4059, s4060, s4061, 
            s4062, s4063, s4064, s4065, s4066, s4067, s4068, s4069, 
            s4070, s4071, s4072, s4073, s4074, s4075, s4076, s4077, 
            s4078, s4079, s4080, s4081, s4082, s4083, s4084, s4085, 
            s4086, s4087, s4088, s4089, s4090, s4091, s4092, s4093, 
            s4094, s4095, s4096, s4097, s4098, s4099, s4100, s4101, 
            s4102, s4103, s4104, s4105, s4106, s4107, s4108, s4109, 
            s4110, s4111, s4112, s4113, s4114, s4115, s4116, s4117, 
            s4118, s4119, s4120, s4121, s4122, s4123, s4124, s4125, 
            s4126, s4127, s4128, s4129, s4130, s4131, s4132, s4133, 
            s4134, s4135, s4136, s4137, s4138, s4139, s4140, s4141, 
            s4142, s4143, s4144, s4145, s4146, s4147, s4148, s4149, 
            s4150, s4151, s4152, s4153, s4154, s4155, s4156, s4157, 
            s4158, s4159, s4160, s4161, s4162, s4163, s4164, s4165, 
            s4166, s4167, s4168, s4169, s4170, s4171, s4172, s4173, 
            s4174, s4175, s4176, s4177, s4178, s4179, s4180, s4181, 
            s4182, s4183, s4184, s4185, s4186, s4187, s4188, s4189, 
            s4190, s4191, s4192, s4193, t7130, t7131, t7132, t7133, 
            t7134, t7135, t7136, t7137, t7138, t7139, t7140, t7141, 
            t7142, t7143, t7144, t7145, t7146, t7147, t7148, t7149, 
            t7150, t7151, t7152, t7153, t7154, t7155, t7156, t7157, 
            t7158, t7159, t7160, t7161, t7162, t7163, t7164, t7165, 
            t7166, t7167, t7168, t7169, t7170, t7171, t7172, t7173, 
            t7174, t7175, t7176, t7177, t7178, t7179, t7180, t7181, 
            t7182, t7183, t7184, t7185, t7186, t7187, t7188, t7189, 
            t7190, t7191, t7192, t7193, t7194, t7195, t7196, t7197, 
            t7198, t7199, t7200, t7201, t7202, t7203, t7204, t7205, 
            t7206, t7207, t7208, t7209, t7210, t7211, t7212, t7213, 
            t7214, t7215, t7216, t7217, t7218, t7219, t7220, t7221, 
            t7222, t7223, t7224, t7225, t7226, t7227, t7228, t7229, 
            t7230, t7231, t7232, t7233, t7234, t7235, t7236, t7237, 
            t7238, t7239, t7240, t7241, t7242, t7243, t7244, t7245, 
            t7246, t7247, t7248, t7249, t7250, t7251, t7252, t7253, 
            t7254, t7255, t7256, t7257, t7258, t7259, t7260, t7261, 
            t7262, t7263, t7264, t7265, t7266, t7267, t7268, t7269, 
            t7270, t7271, t7272, t7273, t7274, t7275, t7276, t7277, 
            t7278, t7279, t7280, t7281, t7282, t7283, t7284, t7285, 
            t7286, t7287, t7288, t7289, t7290, t7291, t7292, t7293, 
            t7294, t7295, t7296, t7297, t7298, t7299, t7300, t7301, 
            t7302, t7303, t7304, t7305, t7306, t7307, t7308, t7309, 
            t7310, t7311, t7312, t7313, t7314, t7315, t7316, t7317, 
            t7318, t7319, t7320, t7321, t7322, t7323, t7324, t7325, 
            t7326, t7327, t7328, t7329, t7330, t7331, t7332, t7333, 
            t7334, t7335, t7336, t7337, t7338, t7339, t7340, t7341, 
            t7342, t7343, t7344, t7345, t7346, t7347, t7348, t7349, 
            t7350, t7351, t7352, t7353, t7354, t7355, t7356, t7357, 
            t7358, t7359, t7360, t7361, t7362, t7363, t7364, t7365, 
            t7366, t7367, t7368, t7369;
    int a10892, a10893, a10894, a10895, a10896, a10897, a10898, a10899, 
            a10900, a10901, a10902, a10903, a10904, a10905, a10906, a10907, 
            a10908, a10909, a10910, a10911, a10912, a10913, a10914, a10915, 
            a10916, a10917, a10918, a10919, a10920, a10921, a10922, a10923, 
            a10924, a10925, a10942, a11005, a11006, a11007, a11008, a11009, 
            a11010, a11011, a11012, a11013, a11014, a11015, a11016, a11017, 
            a11018, a11019, a11020, a11021, a11022, a11023, a11024, a11025, 
            a11026, a11027, a11028, a11029, a11030, a11031, a11032, a11033, 
            a11034, a11035, a11036, a11037, b481, j1, l1, m1;
    l1 = *(lp1);
    m1 = *(mp1);
    for(int j2 = 0; j2 < (l1 - 1); j2++) {
        j1 = (j2 + 1);
        for(int k1 = 0; k1 < m1; k1++) {
            a10892 = (2*k1);
            a10893 = (j1*m1);
            a10894 = (a10892 + (2*a10893));
            s3988 = X[a10894];
            s3989 = X[(a10894 + 1)];
            b481 = (l1*m1);
            a10895 = (a10894 + (2*b481));
            s3990 = X[a10895];
            s3991 = X[(a10895 + 1)];
            a10896 = (a10894 + (4*b481));
            s3992 = X[a10896];
            s3993 = X[(a10896 + 1)];
            a10897 = (a10894 + (6*b481));
            s3994 = X[a10897];
            s3995 = X[(a10897 + 1)];
            a10898 = (a10894 + (8*b481));
            s3996 = X[a10898];
            s3997 = X[(a10898 + 1)];
            a10899 = (a10894 + (10*b481));
            s3998 = X[a10899];
            s3999 = X[(a10899 + 1)];
            a10900 = (a10894 + (12*b481));
            s4000 = X[a10900];
            s4001 = X[(a10900 + 1)];
            a10901 = (a10894 + (14*b481));
            s4002 = X[a10901];
            s4003 = X[(a10901 + 1)];
            a10902 = (a10894 + (16*b481));
            s4004 = X[a10902];
            s4005 = X[(a10902 + 1)];
            a10903 = (a10894 + (18*b481));
            s4006 = X[a10903];
            s4007 = X[(a10903 + 1)];
            a10904 = (a10894 + (20*b481));
            s4008 = X[a10904];
            s4009 = X[(a10904 + 1)];
            a10905 = (a10894 + (22*b481));
            s4010 = X[a10905];
            s4011 = X[(a10905 + 1)];
            a10906 = (a10894 + (24*b481));
            s4012 = X[a10906];
            s4013 = X[(a10906 + 1)];
            a10907 = (a10894 + (26*b481));
            s4014 = X[a10907];
            s4015 = X[(a10907 + 1)];
            a10908 = (a10894 + (28*b481));
            s4016 = X[a10908];
            s4017 = X[(a10908 + 1)];
            a10909 = (a10894 + (30*b481));
            s4018 = X[a10909];
            s4019 = X[(a10909 + 1)];
            a10910 = (a10894 + (32*b481));
            s4020 = X[a10910];
            s4021 = X[(a10910 + 1)];
            a10911 = (a10894 + (34*b481));
            s4022 = X[a10911];
            s4023 = X[(a10911 + 1)];
            a10912 = (a10894 + (36*b481));
            s4024 = X[a10912];
            s4025 = X[(a10912 + 1)];
            a10913 = (a10894 + (38*b481));
            s4026 = X[a10913];
            s4027 = X[(a10913 + 1)];
            a10914 = (a10894 + (40*b481));
            s4028 = X[a10914];
            s4029 = X[(a10914 + 1)];
            a10915 = (a10894 + (42*b481));
            s4030 = X[a10915];
            s4031 = X[(a10915 + 1)];
            a10916 = (a10894 + (44*b481));
            s4032 = X[a10916];
            s4033 = X[(a10916 + 1)];
            a10917 = (a10894 + (46*b481));
            s4034 = X[a10917];
            s4035 = X[(a10917 + 1)];
            a10918 = (a10894 + (48*b481));
            s4036 = X[a10918];
            s4037 = X[(a10918 + 1)];
            a10919 = (a10894 + (50*b481));
            s4038 = X[a10919];
            s4039 = X[(a10919 + 1)];
            a10920 = (a10894 + (52*b481));
            s4040 = X[a10920];
            s4041 = X[(a10920 + 1)];
            a10921 = (a10894 + (54*b481));
            s4042 = X[a10921];
            s4043 = X[(a10921 + 1)];
            a10922 = (a10894 + (56*b481));
            s4044 = X[a10922];
            s4045 = X[(a10922 + 1)];
            a10923 = (a10894 + (58*b481));
            s4046 = X[a10923];
            s4047 = X[(a10923 + 1)];
            a10924 = (a10894 + (60*b481));
            s4048 = X[a10924];
            s4049 = X[(a10924 + 1)];
            a10925 = (a10894 + (62*b481));
            s4050 = X[a10925];
            s4051 = X[(a10925 + 1)];
            t7130 = (s3988 + s4020);
            t7131 = (s3989 + s4021);
            t7132 = (s3988 - s4020);
            t7133 = (s3989 - s4021);
            t7134 = (s4004 + s4036);
            t7135 = (s4005 + s4037);
            t7136 = (s4004 - s4036);
            t7137 = (s4005 - s4037);
            t7138 = (t7130 + t7134);
            t7139 = (t7131 + t7135);
            t7140 = (t7130 - t7134);
            t7141 = (t7131 - t7135);
            s4052 = ((D15[0]*t7138) - (D15[1]*t7139));
            s4053 = ((D15[1]*t7138) + (D15[0]*t7139));
            s4054 = ((D15[2]*t7140) - (D15[3]*t7141));
            s4055 = ((D15[3]*t7140) + (D15[2]*t7141));
            t7142 = (t7132 + t7137);
            t7143 = (t7133 - t7136);
            t7144 = (t7132 - t7137);
            t7145 = (t7133 + t7136);
            s4056 = ((D15[4]*t7142) - (D15[5]*t7143));
            s4057 = ((D15[5]*t7142) + (D15[4]*t7143));
            s4058 = ((D15[6]*t7144) - (D15[7]*t7145));
            s4059 = ((D15[7]*t7144) + (D15[6]*t7145));
            t7146 = (s3990 + s4022);
            t7147 = (s3991 + s4023);
            t7148 = (s3990 - s4022);
            t7149 = (s3991 - s4023);
            t7150 = (s4006 + s4038);
            t7151 = (s4007 + s4039);
            t7152 = (s4006 - s4038);
            t7153 = (s4007 - s4039);
            t7154 = (t7146 + t7150);
            t7155 = (t7147 + t7151);
            t7156 = (t7146 - t7150);
            t7157 = (t7147 - t7151);
            s4060 = ((D15[8]*t7154) - (D15[9]*t7155));
            s4061 = ((D15[9]*t7154) + (D15[8]*t7155));
            s4062 = ((D15[10]*t7156) - (D15[11]*t7157));
            s4063 = ((D15[11]*t7156) + (D15[10]*t7157));
            t7158 = (t7148 + t7153);
            t7159 = (t7149 - t7152);
            t7160 = (t7148 - t7153);
            t7161 = (t7149 + t7152);
            s4064 = ((D15[12]*t7158) - (D15[13]*t7159));
            s4065 = ((D15[13]*t7158) + (D15[12]*t7159));
            s4066 = ((D15[14]*t7160) - (D15[15]*t7161));
            s4067 = ((D15[15]*t7160) + (D15[14]*t7161));
            t7162 = (s3992 + s4024);
            t7163 = (s3993 + s4025);
            t7164 = (s3992 - s4024);
            t7165 = (s3993 - s4025);
            t7166 = (s4008 + s4040);
            t7167 = (s4009 + s4041);
            t7168 = (s4008 - s4040);
            t7169 = (s4009 - s4041);
            t7170 = (t7162 + t7166);
            t7171 = (t7163 + t7167);
            t7172 = (t7162 - t7166);
            t7173 = (t7163 - t7167);
            s4068 = ((D15[16]*t7170) - (D15[17]*t7171));
            s4069 = ((D15[17]*t7170) + (D15[16]*t7171));
            s4070 = ((D15[18]*t7172) - (D15[19]*t7173));
            s4071 = ((D15[19]*t7172) + (D15[18]*t7173));
            t7174 = (t7164 + t7169);
            t7175 = (t7165 - t7168);
            t7176 = (t7164 - t7169);
            t7177 = (t7165 + t7168);
            s4072 = ((D15[20]*t7174) - (D15[21]*t7175));
            s4073 = ((D15[21]*t7174) + (D15[20]*t7175));
            s4074 = ((D15[22]*t7176) - (D15[23]*t7177));
            s4075 = ((D15[23]*t7176) + (D15[22]*t7177));
            t7178 = (s3994 + s4026);
            t7179 = (s3995 + s4027);
            t7180 = (s3994 - s4026);
            t7181 = (s3995 - s4027);
            t7182 = (s4010 + s4042);
            t7183 = (s4011 + s4043);
            t7184 = (s4010 - s4042);
            t7185 = (s4011 - s4043);
            t7186 = (t7178 + t7182);
            t7187 = (t7179 + t7183);
            t7188 = (t7178 - t7182);
            t7189 = (t7179 - t7183);
            s4076 = ((D15[24]*t7186) - (D15[25]*t7187));
            s4077 = ((D15[25]*t7186) + (D15[24]*t7187));
            s4078 = ((D15[26]*t7188) - (D15[27]*t7189));
            s4079 = ((D15[27]*t7188) + (D15[26]*t7189));
            t7190 = (t7180 + t7185);
            t7191 = (t7181 - t7184);
            t7192 = (t7180 - t7185);
            t7193 = (t7181 + t7184);
            s4080 = ((D15[28]*t7190) - (D15[29]*t7191));
            s4081 = ((D15[29]*t7190) + (D15[28]*t7191));
            s4082 = ((D15[30]*t7192) - (D15[31]*t7193));
            s4083 = ((D15[31]*t7192) + (D15[30]*t7193));
            t7194 = (s3996 + s4028);
            t7195 = (s3997 + s4029);
            t7196 = (s3996 - s4028);
            t7197 = (s3997 - s4029);
            t7198 = (s4012 + s4044);
            t7199 = (s4013 + s4045);
            t7200 = (s4012 - s4044);
            t7201 = (s4013 - s4045);
            t7202 = (t7194 + t7198);
            t7203 = (t7195 + t7199);
            t7204 = (t7194 - t7198);
            t7205 = (t7195 - t7199);
            s4084 = ((D15[32]*t7202) - (D15[33]*t7203));
            s4085 = ((D15[33]*t7202) + (D15[32]*t7203));
            s4086 = ((D15[34]*t7204) - (D15[35]*t7205));
            s4087 = ((D15[35]*t7204) + (D15[34]*t7205));
            t7206 = (t7196 + t7201);
            t7207 = (t7197 - t7200);
            t7208 = (t7196 - t7201);
            t7209 = (t7197 + t7200);
            s4088 = ((D15[36]*t7206) - (D15[37]*t7207));
            s4089 = ((D15[37]*t7206) + (D15[36]*t7207));
            s4090 = ((D15[38]*t7208) - (D15[39]*t7209));
            s4091 = ((D15[39]*t7208) + (D15[38]*t7209));
            t7210 = (s3998 + s4030);
            t7211 = (s3999 + s4031);
            t7212 = (s3998 - s4030);
            t7213 = (s3999 - s4031);
            t7214 = (s4014 + s4046);
            t7215 = (s4015 + s4047);
            t7216 = (s4014 - s4046);
            t7217 = (s4015 - s4047);
            t7218 = (t7210 + t7214);
            t7219 = (t7211 + t7215);
            t7220 = (t7210 - t7214);
            t7221 = (t7211 - t7215);
            s4092 = ((D15[40]*t7218) - (D15[41]*t7219));
            s4093 = ((D15[41]*t7218) + (D15[40]*t7219));
            s4094 = ((D15[42]*t7220) - (D15[43]*t7221));
            s4095 = ((D15[43]*t7220) + (D15[42]*t7221));
            t7222 = (t7212 + t7217);
            t7223 = (t7213 - t7216);
            t7224 = (t7212 - t7217);
            t7225 = (t7213 + t7216);
            s4096 = ((D15[44]*t7222) - (D15[45]*t7223));
            s4097 = ((D15[45]*t7222) + (D15[44]*t7223));
            s4098 = ((D15[46]*t7224) - (D15[47]*t7225));
            s4099 = ((D15[47]*t7224) + (D15[46]*t7225));
            t7226 = (s4000 + s4032);
            t7227 = (s4001 + s4033);
            t7228 = (s4000 - s4032);
            t7229 = (s4001 - s4033);
            t7230 = (s4016 + s4048);
            t7231 = (s4017 + s4049);
            t7232 = (s4016 - s4048);
            t7233 = (s4017 - s4049);
            t7234 = (t7226 + t7230);
            t7235 = (t7227 + t7231);
            t7236 = (t7226 - t7230);
            t7237 = (t7227 - t7231);
            s4100 = ((D15[48]*t7234) - (D15[49]*t7235));
            s4101 = ((D15[49]*t7234) + (D15[48]*t7235));
            s4102 = ((D15[50]*t7236) - (D15[51]*t7237));
            s4103 = ((D15[51]*t7236) + (D15[50]*t7237));
            t7238 = (t7228 + t7233);
            t7239 = (t7229 - t7232);
            t7240 = (t7228 - t7233);
            t7241 = (t7229 + t7232);
            s4104 = ((D15[52]*t7238) - (D15[53]*t7239));
            s4105 = ((D15[53]*t7238) + (D15[52]*t7239));
            s4106 = ((D15[54]*t7240) - (D15[55]*t7241));
            s4107 = ((D15[55]*t7240) + (D15[54]*t7241));
            t7242 = (s4002 + s4034);
            t7243 = (s4003 + s4035);
            t7244 = (s4002 - s4034);
            t7245 = (s4003 - s4035);
            t7246 = (s4018 + s4050);
            t7247 = (s4019 + s4051);
            t7248 = (s4018 - s4050);
            t7249 = (s4019 - s4051);
            t7250 = (t7242 + t7246);
            t7251 = (t7243 + t7247);
            t7252 = (t7242 - t7246);
            t7253 = (t7243 - t7247);
            s4108 = ((D15[56]*t7250) - (D15[57]*t7251));
            s4109 = ((D15[57]*t7250) + (D15[56]*t7251));
            s4110 = ((D15[58]*t7252) - (D15[59]*t7253));
            s4111 = ((D15[59]*t7252) + (D15[58]*t7253));
            t7254 = (t7244 + t7249);
            t7255 = (t7245 - t7248);
            t7256 = (t7244 - t7249);
            t7257 = (t7245 + t7248);
            s4112 = ((D15[60]*t7254) - (D15[61]*t7255));
            s4113 = ((D15[61]*t7254) + (D15[60]*t7255));
            s4114 = ((D15[62]*t7256) - (D15[63]*t7257));
            s4115 = ((D15[63]*t7256) + (D15[62]*t7257));
            t7258 = (s4052 + s4084);
            t7259 = (s4053 + s4085);
            t7260 = (s4052 - s4084);
            t7261 = (s4053 - s4085);
            t7262 = (s4068 + s4100);
            t7263 = (s4069 + s4101);
            t7264 = (s4068 - s4100);
            t7265 = (s4069 - s4101);
            t7266 = (t7258 + t7262);
            t7267 = (t7259 + t7263);
            t7268 = (t7258 - t7262);
            t7269 = (t7259 - t7263);
            t7270 = (t7260 + t7265);
            t7271 = (t7261 - t7264);
            t7272 = (t7260 - t7265);
            t7273 = (t7261 + t7264);
            t7274 = (s4060 + s4092);
            t7275 = (s4061 + s4093);
            t7276 = (s4060 - s4092);
            t7277 = (s4061 - s4093);
            t7278 = (s4076 + s4108);
            t7279 = (s4077 + s4109);
            t7280 = (s4076 - s4108);
            t7281 = (s4077 - s4109);
            t7282 = (t7274 + t7278);
            t7283 = (t7275 + t7279);
            t7284 = (t7274 - t7278);
            t7285 = (t7275 - t7279);
            a10926 = (0.70710678118654757*(t7276 + t7281));
            a10927 = (0.70710678118654757*(t7277 - t7280));
            s4116 = (a10926 + a10927);
            s4117 = (a10927 - a10926);
            a10928 = (0.70710678118654757*(t7277 + t7280));
            a10929 = (0.70710678118654757*(t7276 - t7281));
            s4118 = (a10928 - a10929);
            s4119 = (a10929 + a10928);
            s4120 = (t7266 - t7282);
            s4121 = (t7267 - t7283);
            s4122 = (t7270 + s4116);
            s4123 = (t7271 + s4117);
            s4124 = (t7270 - s4116);
            s4125 = (t7271 - s4117);
            s4126 = (t7268 + t7285);
            s4127 = (t7269 - t7284);
            s4128 = (t7268 - t7285);
            s4129 = (t7269 + t7284);
            s4130 = (t7272 + s4118);
            s4131 = (t7273 - s4119);
            s4132 = (t7272 - s4118);
            s4133 = (t7273 + s4119);
            t7286 = (s4056 + s4088);
            t7287 = (s4057 + s4089);
            t7288 = (s4056 - s4088);
            t7289 = (s4057 - s4089);
            t7290 = (s4072 + s4104);
            t7291 = (s4073 + s4105);
            t7292 = (s4072 - s4104);
            t7293 = (s4073 - s4105);
            t7294 = (t7286 + t7290);
            t7295 = (t7287 + t7291);
            t7296 = (t7286 - t7290);
            t7297 = (t7287 - t7291);
            t7298 = (t7288 + t7293);
            t7299 = (t7289 - t7292);
            t7300 = (t7288 - t7293);
            t7301 = (t7289 + t7292);
            t7302 = (s4064 + s4096);
            t7303 = (s4065 + s4097);
            t7304 = (s4064 - s4096);
            t7305 = (s4065 - s4097);
            t7306 = (s4080 + s4112);
            t7307 = (s4081 + s4113);
            t7308 = (s4080 - s4112);
            t7309 = (s4081 - s4113);
            t7310 = (t7302 + t7306);
            t7311 = (t7303 + t7307);
            t7312 = (t7302 - t7306);
            t7313 = (t7303 - t7307);
            a10930 = (0.70710678118654757*(t7304 + t7309));
            a10931 = (0.70710678118654757*(t7305 - t7308));
            s4134 = (a10930 + a10931);
            s4135 = (a10931 - a10930);
            a10932 = (0.70710678118654757*(t7305 + t7308));
            a10933 = (0.70710678118654757*(t7304 - t7309));
            s4136 = (a10932 - a10933);
            s4137 = (a10933 + a10932);
            s4138 = (t7294 + t7310);
            s4139 = (t7295 + t7311);
            s4140 = (t7294 - t7310);
            s4141 = (t7295 - t7311);
            s4142 = (t7298 + s4134);
            s4143 = (t7299 + s4135);
            s4144 = (t7298 - s4134);
            s4145 = (t7299 - s4135);
            s4146 = (t7296 + t7313);
            s4147 = (t7297 - t7312);
            s4148 = (t7296 - t7313);
            s4149 = (t7297 + t7312);
            s4150 = (t7300 + s4136);
            s4151 = (t7301 - s4137);
            s4152 = (t7300 - s4136);
            s4153 = (t7301 + s4137);
            t7314 = (s4054 + s4086);
            t7315 = (s4055 + s4087);
            t7316 = (s4054 - s4086);
            t7317 = (s4055 - s4087);
            t7318 = (s4070 + s4102);
            t7319 = (s4071 + s4103);
            t7320 = (s4070 - s4102);
            t7321 = (s4071 - s4103);
            t7322 = (t7314 + t7318);
            t7323 = (t7315 + t7319);
            t7324 = (t7314 - t7318);
            t7325 = (t7315 - t7319);
            t7326 = (t7316 + t7321);
            t7327 = (t7317 - t7320);
            t7328 = (t7316 - t7321);
            t7329 = (t7317 + t7320);
            t7330 = (s4062 + s4094);
            t7331 = (s4063 + s4095);
            t7332 = (s4062 - s4094);
            t7333 = (s4063 - s4095);
            t7334 = (s4078 + s4110);
            t7335 = (s4079 + s4111);
            t7336 = (s4078 - s4110);
            t7337 = (s4079 - s4111);
            t7338 = (t7330 + t7334);
            t7339 = (t7331 + t7335);
            t7340 = (t7330 - t7334);
            t7341 = (t7331 - t7335);
            a10934 = (0.70710678118654757*(t7332 + t7337));
            a10935 = (0.70710678118654757*(t7333 - t7336));
            s4154 = (a10934 + a10935);
            s4155 = (a10935 - a10934);
            a10936 = (0.70710678118654757*(t7333 + t7336));
            a10937 = (0.70710678118654757*(t7332 - t7337));
            s4156 = (a10936 - a10937);
            s4157 = (a10937 + a10936);
            s4158 = (t7322 + t7338);
            s4159 = (t7323 + t7339);
            s4160 = (t7322 - t7338);
            s4161 = (t7323 - t7339);
            s4162 = (t7326 + s4154);
            s4163 = (t7327 + s4155);
            s4164 = (t7326 - s4154);
            s4165 = (t7327 - s4155);
            s4166 = (t7324 + t7341);
            s4167 = (t7325 - t7340);
            s4168 = (t7324 - t7341);
            s4169 = (t7325 + t7340);
            s4170 = (t7328 + s4156);
            s4171 = (t7329 - s4157);
            s4172 = (t7328 - s4156);
            s4173 = (t7329 + s4157);
            t7342 = (s4058 + s4090);
            t7343 = (s4059 + s4091);
            t7344 = (s4058 - s4090);
            t7345 = (s4059 - s4091);
            t7346 = (s4074 + s4106);
            t7347 = (s4075 + s4107);
            t7348 = (s4074 - s4106);
            t7349 = (s4075 - s4107);
            t7350 = (t7342 + t7346);
            t7351 = (t7343 + t7347);
            t7352 = (t7342 - t7346);
            t7353 = (t7343 - t7347);
            t7354 = (t7344 + t7349);
            t7355 = (t7345 - t7348);
            t7356 = (t7344 - t7349);
            t7357 = (t7345 + t7348);
            t7358 = (s4066 + s4098);
            t7359 = (s4067 + s4099);
            t7360 = (s4066 - s4098);
            t7361 = (s4067 - s4099);
            t7362 = (s4082 + s4114);
            t7363 = (s4083 + s4115);
            t7364 = (s4082 - s4114);
            t7365 = (s4083 - s4115);
            t7366 = (t7358 + t7362);
            t7367 = (t7359 + t7363);
            t7368 = (t7358 - t7362);
            t7369 = (t7359 - t7363);
            a10938 = (0.70710678118654757*(t7360 + t7365));
            a10939 = (0.70710678118654757*(t7361 - t7364));
            s4174 = (a10938 + a10939);
            s4175 = (a10939 - a10938);
            a10940 = (0.70710678118654757*(t7361 + t7364));
            a10941 = (0.70710678118654757*(t7360 - t7365));
            s4176 = (a10940 - a10941);
            s4177 = (a10941 + a10940);
            s4178 = (t7350 + t7366);
            s4179 = (t7351 + t7367);
            s4180 = (t7350 - t7366);
            s4181 = (t7351 - t7367);
            s4182 = (t7354 + s4174);
            s4183 = (t7355 + s4175);
            s4184 = (t7354 - s4174);
            s4185 = (t7355 - s4175);
            s4186 = (t7352 + t7369);
            s4187 = (t7353 - t7368);
            s4188 = (t7352 - t7369);
            s4189 = (t7353 + t7368);
            s4190 = (t7356 + s4176);
            s4191 = (t7357 - s4177);
            s4192 = (t7356 - s4176);
            s4193 = (t7357 + s4177);
            a10942 = (62*j1);
            a10943 = TW1[a10942];
            a10944 = TW1[(a10942 + 1)];
            a10945 = TW1[(a10942 + 2)];
            a10946 = TW1[(a10942 + 3)];
            a10947 = TW1[(a10942 + 4)];
            a10948 = TW1[(a10942 + 5)];
            a10949 = TW1[(a10942 + 6)];
            a10950 = TW1[(a10942 + 7)];
            a10951 = TW1[(a10942 + 8)];
            a10952 = TW1[(a10942 + 9)];
            a10953 = TW1[(a10942 + 10)];
            a10954 = TW1[(a10942 + 11)];
            a10955 = TW1[(a10942 + 12)];
            a10956 = TW1[(a10942 + 13)];
            a10957 = TW1[(a10942 + 14)];
            a10958 = TW1[(a10942 + 15)];
            a10959 = TW1[(a10942 + 16)];
            a10960 = TW1[(a10942 + 17)];
            a10961 = TW1[(a10942 + 18)];
            a10962 = TW1[(a10942 + 19)];
            a10963 = TW1[(a10942 + 20)];
            a10964 = TW1[(a10942 + 21)];
            a10965 = TW1[(a10942 + 22)];
            a10966 = TW1[(a10942 + 23)];
            a10967 = TW1[(a10942 + 24)];
            a10968 = TW1[(a10942 + 25)];
            a10969 = TW1[(a10942 + 26)];
            a10970 = TW1[(a10942 + 27)];
            a10971 = TW1[(a10942 + 28)];
            a10972 = TW1[(a10942 + 29)];
            a10973 = TW1[(a10942 + 30)];
            a10974 = TW1[(a10942 + 31)];
            a10975 = TW1[(a10942 + 32)];
            a10976 = TW1[(a10942 + 33)];
            a10977 = TW1[(a10942 + 34)];
            a10978 = TW1[(a10942 + 35)];
            a10979 = TW1[(a10942 + 36)];
            a10980 = TW1[(a10942 + 37)];
            a10981 = TW1[(a10942 + 38)];
            a10982 = TW1[(a10942 + 39)];
            a10983 = TW1[(a10942 + 40)];
            a10984 = TW1[(a10942 + 41)];
            a10985 = TW1[(a10942 + 42)];
            a10986 = TW1[(a10942 + 43)];
            a10987 = TW1[(a10942 + 44)];
            a10988 = TW1[(a10942 + 45)];
            a10989 = TW1[(a10942 + 46)];
            a10990 = TW1[(a10942 + 47)];
            a10991 = TW1[(a10942 + 48)];
            a10992 = TW1[(a10942 + 49)];
            a10993 = TW1[(a10942 + 50)];
            a10994 = TW1[(a10942 + 51)];
            a10995 = TW1[(a10942 + 52)];
            a10996 = TW1[(a10942 + 53)];
            a10997 = TW1[(a10942 + 54)];
            a10998 = TW1[(a10942 + 55)];
            a10999 = TW1[(a10942 + 56)];
            a11000 = TW1[(a10942 + 57)];
            a11001 = TW1[(a10942 + 58)];
            a11002 = TW1[(a10942 + 59)];
            a11003 = TW1[(a10942 + 60)];
            a11004 = TW1[(a10942 + 61)];
            a11005 = (64*a10893);
            a11006 = (a10892 + a11005);
            Y[a11006] = (t7266 + t7282);
            Y[(a11006 + 1)] = (t7267 + t7283);
            a11007 = (a10892 + (2*m1) + a11005);
            Y[a11007] = ((a10943*s4138) - (a10944*s4139));
            Y[(a11007 + 1)] = ((a10944*s4138) + (a10943*s4139));
            a11008 = (a10892 + (4*m1) + a11005);
            Y[a11008] = ((a10945*s4158) - (a10946*s4159));
            Y[(a11008 + 1)] = ((a10946*s4158) + (a10945*s4159));
            a11009 = (a10892 + (6*m1) + a11005);
            Y[a11009] = ((a10947*s4178) - (a10948*s4179));
            Y[(a11009 + 1)] = ((a10948*s4178) + (a10947*s4179));
            a11010 = (a10892 + (8*m1) + a11005);
            Y[a11010] = ((a10949*s4122) - (a10950*s4123));
            Y[(a11010 + 1)] = ((a10950*s4122) + (a10949*s4123));
            a11011 = (a10892 + (10*m1) + a11005);
            Y[a11011] = ((a10951*s4142) - (a10952*s4143));
            Y[(a11011 + 1)] = ((a10952*s4142) + (a10951*s4143));
            a11012 = (a10892 + (12*m1) + a11005);
            Y[a11012] = ((a10953*s4162) - (a10954*s4163));
            Y[(a11012 + 1)] = ((a10954*s4162) + (a10953*s4163));
            a11013 = (a10892 + (14*m1) + a11005);
            Y[a11013] = ((a10955*s4182) - (a10956*s4183));
            Y[(a11013 + 1)] = ((a10956*s4182) + (a10955*s4183));
            a11014 = (a10892 + (16*m1) + a11005);
            Y[a11014] = ((a10957*s4126) - (a10958*s4127));
            Y[(a11014 + 1)] = ((a10958*s4126) + (a10957*s4127));
            a11015 = (a10892 + (18*m1) + a11005);
            Y[a11015] = ((a10959*s4146) - (a10960*s4147));
            Y[(a11015 + 1)] = ((a10960*s4146) + (a10959*s4147));
            a11016 = (a10892 + (20*m1) + a11005);
            Y[a11016] = ((a10961*s4166) - (a10962*s4167));
            Y[(a11016 + 1)] = ((a10962*s4166) + (a10961*s4167));
            a11017 = (a10892 + (22*m1) + a11005);
            Y[a11017] = ((a10963*s4186) - (a10964*s4187));
            Y[(a11017 + 1)] = ((a10964*s4186) + (a10963*s4187));
            a11018 = (a10892 + (24*m1) + a11005);
            Y[a11018] = ((a10965*s4130) - (a10966*s4131));
            Y[(a11018 + 1)] = ((a10966*s4130) + (a10965*s4131));
            a11019 = (a10892 + (26*m1) + a11005);
            Y[a11019] = ((a10967*s4150) - (a10968*s4151));
            Y[(a11019 + 1)] = ((a10968*s4150) + (a10967*s4151));
            a11020 = (a10892 + (28*m1) + a11005);
            Y[a11020] = ((a10969*s4170) - (a10970*s4171));
            Y[(a11020 + 1)] = ((a10970*s4170) + (a10969*s4171));
            a11021 = (a10892 + (30*m1) + a11005);
            Y[a11021] = ((a10971*s4190) - (a10972*s4191));
            Y[(a11021 + 1)] = ((a10972*s4190) + (a10971*s4191));
            a11022 = (a10892 + (32*m1) + a11005);
            Y[a11022] = ((a10973*s4120) - (a10974*s4121));
            Y[(a11022 + 1)] = ((a10974*s4120) + (a10973*s4121));
            a11023 = (a10892 + (34*m1) + a11005);
            Y[a11023] = ((a10975*s4140) - (a10976*s4141));
            Y[(a11023 + 1)] = ((a10976*s4140) + (a10975*s4141));
            a11024 = (a10892 + (36*m1) + a11005);
            Y[a11024] = ((a10977*s4160) - (a10978*s4161));
            Y[(a11024 + 1)] = ((a10978*s4160) + (a10977*s4161));
            a11025 = (a10892 + (38*m1) + a11005);
            Y[a11025] = ((a10979*s4180) - (a10980*s4181));
            Y[(a11025 + 1)] = ((a10980*s4180) + (a10979*s4181));
            a11026 = (a10892 + (40*m1) + a11005);
            Y[a11026] = ((a10981*s4124) - (a10982*s4125));
            Y[(a11026 + 1)] = ((a10982*s4124) + (a10981*s4125));
            a11027 = (a10892 + (42*m1) + a11005);
            Y[a11027] = ((a10983*s4144) - (a10984*s4145));
            Y[(a11027 + 1)] = ((a10984*s4144) + (a10983*s4145));
            a11028 = (a10892 + (44*m1) + a11005);
            Y[a11028] = ((a10985*s4164) - (a10986*s4165));
            Y[(a11028 + 1)] = ((a10986*s4164) + (a10985*s4165));
            a11029 = (a10892 + (46*m1) + a11005);
            Y[a11029] = ((a10987*s4184) - (a10988*s4185));
            Y[(a11029 + 1)] = ((a10988*s4184) + (a10987*s4185));
            a11030 = (a10892 + (48*m1) + a11005);
            Y[a11030] = ((a10989*s4128) - (a10990*s4129));
            Y[(a11030 + 1)] = ((a10990*s4128) + (a10989*s4129));
            a11031 = (a10892 + (50*m1) + a11005);
            Y[a11031] = ((a10991*s4148) - (a10992*s4149));
            Y[(a11031 + 1)] = ((a10992*s4148) + (a10991*s4149));
            a11032 = (a10892 + (52*m1) + a11005);
            Y[a11032] = ((a10993*s4168) - (a10994*s4169));
            Y[(a11032 + 1)] = ((a10994*s4168) + (a10993*s4169));
            a11033 = (a10892 + (54*m1) + a11005);
            Y[a11033] = ((a10995*s4188) - (a10996*s4189));
            Y[(a11033 + 1)] = ((a10996*s4188) + (a10995*s4189));
            a11034 = (a10892 + (56*m1) + a11005);
            Y[a11034] = ((a10997*s4132) - (a10998*s4133));
            Y[(a11034 + 1)] = ((a10998*s4132) + (a10997*s4133));
            a11035 = (a10892 + (58*m1) + a11005);
            Y[a11035] = ((a10999*s4152) - (a11000*s4153));
            Y[(a11035 + 1)] = ((a11000*s4152) + (a10999*s4153));
            a11036 = (a10892 + (60*m1) + a11005);
            Y[a11036] = ((a11001*s4172) - (a11002*s4173));
            Y[(a11036 + 1)] = ((a11002*s4172) + (a11001*s4173));
            a11037 = (a10892 + (62*m1) + a11005);
            Y[a11037] = ((a11003*s4192) - (a11004*s4193));
            Y[(a11037 + 1)] = ((a11004*s4192) + (a11003*s4193));
        }
    }
}
