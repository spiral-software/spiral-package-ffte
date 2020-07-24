# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# Radix N kernel for FFTE

buildScalarKernel_c := function(n, kk, conf, opts)
    local name, suffix, filesuffix, j, k, l, l1, m, tmp1, tmp3, i1, i2, gf, sf, gath, scat, gc, sc, 
        dft, rts, opcnts, rt, dfts, dftc, cl, cc, c, clp, lp, mp;

    name := DFTnameStr(n, kk);
    suffix := "c";
    filesuffix := "c_";
#    suffix := "_k";
    Print("\n\n// == Generating \"", name, "\" ==================================================\n\n");    

    # variables
    j := V(0);
    k := var.fresh_t("k", TInt);
    l := V(1);
    l1 := var.fresh_t("l", TInt);
    m := var.fresh_t("m", TInt);
    lp := var.fresh_t("lp", TPtr(TInt));
    mp := var.fresh_t("mp", TPtr(TInt));
    
    # temp arrays
    tmp1 := var.fresh_t("R", TArray(TReal, 2*n));
    tmp3 := var.fresh_t("T", TArray(TReal, 2*n));
    
    # gather and scatter
    i1 := var.fresh_t("i", n);
    i2 := var.fresh_t("i", n);
    gf := fTensor(Lambda(i1, k + j*m + i1*l1*m), fId(2));
    sf := fTensor(Lambda(i2, k+n*j*m + i2*m), fId(2));
    gath := Gath(gf);
    scat := Scat(sf);
    
    # generate gather and scatter code
    gc := BlockUnroll(unroll_cmd(opts.codegen(gath, tmp1, X, opts)), opts);
    sc := BlockUnroll(unroll_cmd(opts.codegen(scat, Y, tmp3, opts)), opts);

    # generate DFT kernel
    dft := RC(DFT(n, kk));
    rts := AllRuleTrees(dft, opts);
    opcnts := List(rts, r -> [ Length(Collect(CodeRuleTree(r, opts), @(1, [add, sub, mul], e->e.t=TReal))), r]);
    rt := Minimum(opcnts)[2];
    dfts := SumsRuleTree(rt, opts);
    dftc := BlockUnroll(opts.codegen(dfts, tmp3, tmp1, opts), opts);

    # stitch code fragments together
    cl := func(TVoid, "transform", [Y, X, j, k, l1, m], 
            decl([tmp1, tmp3], chain(gc, dftc, sc)));
    cc := Compile(cl, opts);
    
    # fixup to push decls inside (?)
    c := func(TVoid, "transform", [Y, X, j, k, l1, m], 
            decl(cc.vars, cc.cmd.cmds[1].cmd));
    
    # print Radix N FFTE kernel as function
    
    # function with j and k loop in the function call
    clp := func(TVoid, "transform", [Y, X, lp, mp], 
        decl(c.cmd.vars::c.free()::[l1,m],
            chain(
                assign(l1, deref(lp)),
                assign(m, deref(mp)),
                loopn(k, m, c.cmd.cmd)
            )
        )
    );
    
    # print Radix N FFTE kernel as function
    PrintCode(opts.FortranIze(name::suffix), clp, opts);
    PrintTo(name::filesuffix::suffix::".c", PrintCode(opts.FortranIze(name::suffix), clp, opts));
end;
