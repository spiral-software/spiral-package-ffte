# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# This is the main FFTE Kernel generation file
# Run as "spiral < ffte.g" or
# copy into Spiral command prompt

LoadImport(ffte);
Import(ffte.kernelgen);

sve_opts := FFTE_SVEopts;
c_opts := FFTEopts;

conf := rec(
    buildScalar := true,
    buildSVE := true,
    useFMA := true,
    test := false
);

if conf.test then
    kernelSizes := [2];
    kList := [-1];
else
    kernelSizes := Filtered([2..32], i-> ForAll(Factors(i), j->j in [2,3,5,7]));
    kList := [-1];
fi;

if conf.buildScalar then
    buildKernels(buildScalarKernel_a, kernelSizes, kList, conf, c_opts);
    buildKernels(buildScalarKernel_b, kernelSizes, kList, conf, c_opts);
    buildKernels(buildScalarKernel_c, kernelSizes, kList, conf, c_opts);
fi;

if conf.buildSVE then
    buildKernels(buildSVEKernel_a, kernelSizes, kList, conf, sve_opts);
    buildKernels(buildSVEKernel_b, kernelSizes, kList, conf, sve_opts);
    buildKernels(buildSVEKernel_c, kernelSizes, kList, conf, sve_opts);
fi;

