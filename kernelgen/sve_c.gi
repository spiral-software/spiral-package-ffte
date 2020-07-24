# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# Radix n kernel for FFTE, targeting ARM SVE

buildSVEKernel_c := function(n, kk, conf, opts)
    local vlen, name, suffix, filesuffix, j, k, l, m, tmp1, tmp3, 
        gi, g, gc, si, ss, s, sc, dft, rts, rt, opcnts, dfts, dftc, 
        cl, cc, c, clp, useFMA, _FMA, lp, mp;

    name := DFTnameStr(n, kk);
    suffix := "c";
    filesuffix := "sve_";
#    suffix := "_k";
    Print("\n\n// == Generating \"", name, "\" ==================================================\n\n");    

    vlen := TInt;
    useFMA := When(IsBound(conf.useFMA), conf.useFMA, true);
    _FMA := When(useFMA, FMA, p -> p);

    # variables
    j := V(0);
    k := var.fresh_t("k", TInt);
    l := var.fresh_t("l", TInt);
    #l := V(1);
    m := var.fresh_t("m", TInt);
    lp := var.fresh_t("lp", TPtr(TInt));
    mp := var.fresh_t("mp", TPtr(TInt));
    
    # temp arrays
    tmp1 := var.fresh_t("R", TArray(TVectSVE, 2*n));
    tmp3 := var.fresh_t("T", TArray(TVectSVE, 2*n));
    
    # gather
    gi := _i -> VGath_L(1, vlen, 2*(k + j*m + _i*l*m));
    g := RulesSums(SumsSPL(VStack(List([0..n-1], gi)), opts));
    gc := Compile(opts.codegen(g, tmp1, X, opts), opts);
    
    # scatter
    si := _i -> VScat_L(1, vlen, 2*(k+n*j*m + _i*m));
    ss := SumsSPL(HStack(List([0..n-1], si)), opts);
    s := RulesSums(SubstTopDown(ss, @(1, ScatAcc), e->Scat(@(1).val.func)));
    sc := Compile(opts.codegen(s, Y, tmp3, opts), opts);
    
    # generate DFT kernel
    dft := RC(DFT(n, kk));
    rts := AllRuleTrees(dft, opts);
    opcnts := List(rts, r -> [ Length(Collect(CodeRuleTree(r, opts), @(1, [add, sub, mul], e->e.t=TReal))), r]);
    rt := Minimum(opcnts)[2];
    dfts := SumsRuleTree(rt, opts);
    dftc := BlockUnroll(opts.codegen(dfts, tmp3, tmp1, opts), opts);
    
    # stitch code fragments together
    cl := func(TVoid, "transform", [Y, X, j, k, l, m, opts.pg], 
            decl([tmp1, tmp3], chain(gc, dftc, sc)));
    cc := Compile(cl, opts);
    
    # fixup to push decls inside (?)
    c := func(TVoid, "transform", [Y, X, k, l, m, opts.pg], 
            decl(cc.vars, _FMA(cc.cmd.cmds[1].cmd)));
    
    # print Radix N FFTE kernel as function
    #PrintCode(name, c, opts);
    #PrintTo(name::".c", PrintCode(name, c, opts));
    
    ########################################################
    # version with j and k loop in the function call
    clp := func(TVoid, "transform", [Y, X, lp, mp], 
        decl(c.cmd.vars::c.free()::[opts.pg, l, m], 
            chain(
                assign(l, deref(lp)),
                assign(m, deref(mp)),
                sve_loopn(k, m, c.cmd.cmd)
            )
        )
    );
    
    # print Radix N FFTE kernel as function
    PrintCode(opts.FortranIze(name::suffix), clp, opts);
    PrintTo(name::filesuffix::suffix::".c", PrintCode(opts.FortranIze(name::suffix), clp, opts));
end;




