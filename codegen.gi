# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

Class(SVECodegen, DefaultCodegen, rec(
    GathH :=  ( self, o, y, x, opts ) >> assign(nth(y, 0), sve_gath(nth(x, o.base).toPtr(TVectSVE), o.n, opts.pg, o.stride)),
#    ScatH :=  ( self, o, y, x, opts ) >> sve_scat(nth(y, o.base).toPtr(TVectSVE), o.n, opts.pg, o.stride, nth(x, 0)),
    ScatH :=  ( self, o, y, x, opts ) >> sve_scat(nth(y, o.base).toPtr(TReal), o.n, opts.pg, o.stride, nth(x, 0)),
    
    VGath_L :=  ( self, o, y, x, opts ) >> let(_v := var.fresh_t("svex2_", TVectSVEx2), 
        decl([_v], chain(
            assign(_v, sve_ld2(nth(x, o.base).toPtr(TReal), o.n, opts.pg)),
            assign(nth(y, 0), velem(_v, 0)),
            assign(nth(y, 1), velem(_v, 1))
        )) 
    ),

    VScat_L :=  ( self, o, y, x, opts ) >> let(_v := var.fresh_t("svex2_", TVectSVEx2), 
        decl([_v], chain(
            assign(velem(_v, 0), nth(x, 0)),
            assign(velem(_v, 1), nth(x, 1)),
            sve_st2(nth(y, o.base).toPtr(TReal), o.n, opts.pg, _v)
        ))
    )

));



