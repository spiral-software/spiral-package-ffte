# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

DFTnameStr := (_n, _k) -> When(_k=-1, "dft", "idft")::StringInt(_n);

Include(scalar_a);
Include(scalar_b);
Include(scalar_c);
Include(sve_a);
Include(sve_b);
Include(sve_c);


buildKernels := function(kernelGen, kernelSizes, kList, conf, opts)
    local n, kk;
    for kk in kList do
        for n in kernelSizes do
            var.flush();
            ApplyFunc(kernelGen, [n, kk, conf, opts]);
        od;
    od;
end;


