# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# setup
Include(sigmaspl);
Include(code);
Include(codegen);
Include(unparse);
LoadImport(ffte.kernelgen);

# load copyright
Include(copyright);

# SVE options
Class(SVEopts, SpiralDefaults);
SVEopts.useDeref := false;
SVEopts.pg := var.fresh_t("pg", TBoolSVE);

SVEopts.unparser := SVEUnparser;
SVEopts.codegen := SVECodegen;
SVEopts.TRealCtype := "float64_t";
SVEopts.includes := ["<arm_sve.h>"];

# FFTE options
Class(FFTEopts, SpiralDefaults);
FFTEopts.useDeref := false;
FFTEopts.breakdownRules.DFT := [DFT_Base, DFT_CT, DFT_CT_Mincost, DFT_Rader, DFT_GoodThomas, DFT_SplitRadix ];
FFTEopts.unparser.generated_by := FFTE_copyright;
FFTEopts.FortranIze := s->s::"_";
FFTEopts.includes := [];

Class(FFTE_SVEopts, SVEopts);
FFTE_SVEopts.breakdownRules.DFT := [DFT_Base, DFT_CT, DFT_CT_Mincost, DFT_Rader, DFT_GoodThomas, DFT_SplitRadix ];
FFTE_SVEopts.unparser.generated_by := FFTE_copyright;
FFTE_SVEopts.FortranIze := s->s::"_";

# install type unification "handler"
spiral.code.__UnifyPair := spiral.code.UnifyPair;
spiral.code.UnifyPair := (a, b) -> When(a = TVectSVE or b = TVectSVE, TVectSVE, spiral.code.__UnifyPair(a, b));
