"""Reference fixtures for NumPy's polynomial bases and legacy routines."""
import importlib
import numpy as np


def populate_polynomial(record, fails):
    specs=[("polynomial","poly","Polynomial"),("chebyshev","cheb","Chebyshev"),
           ("legendre","leg","Legendre"),("hermite","herm","Hermite"),
           ("hermite_e","herme","HermiteE"),("laguerre","lag","Laguerre")]
    c=np.array([1.5,-.5,.75,1.2])
    d=np.array([-.3,2.,1.])
    roots=np.array([-.7,.2,.9])
    x=np.array([[-.8,-.2,0.],[.1,.4,.7]])
    y=x+.1
    z=x-.15
    coefND=np.arange(6.,dtype=float).reshape(3,2)/7+.1
    coef3=np.arange(12.,dtype=float).reshape(2,3,2)/9-.2
    gx=np.array([-.5,.3]);gy=np.array([-.3,.2,.8]);gz=np.array([.1,.9])
    fitX=np.linspace(-.9,.9,10);weights=np.linspace(.8,1.2,10)
    record("poly_input_c",c);record("poly_input_d",d);record("poly_input_roots",roots)
    record("poly_input_x",x);record("poly_input_y",y);record("poly_input_z",z)
    record("poly_input_coeff_nd",coefND);record("poly_input_coeff_3",coef3)
    record("poly_input_gx",gx);record("poly_input_gy",gy);record("poly_input_gz",gz)
    record("poly_input_fit_x",fitX);record("poly_input_weights",weights)
    for name,prefix,clsname in specs:
        module=importlib.import_module("numpy.polynomial."+name)
        f=lambda op:getattr(module,prefix+op)
        tag="poly_"+name+"_"
        for op in ["add","sub","mul"]:
            record(tag+op,f(op)(c,d))
        q,r=f("div")(c,d);record(tag+"div_q",q);record(tag+"div_r",r)
        record(tag+"mulx",f("mulx")(c))
        record(tag+"pow",f("pow")(c,3))
        record(tag+"trim",f("trim")([1.,2.,1e-8,0.],tol=1e-7))
        record(tag+"line",f("line")(2.,3.))
        rcoef=f("fromroots")(roots)
        record(tag+"fromroots",rcoef);record(tag+"roots",f("roots")(rcoef))
        complex_roots=np.array([1+2j,-1+.5j])
        record(tag+"complex_roots",f("roots")(f("fromroots")(complex_roots)))
        record(tag+"companion",f("companion")(c))
        record(tag+"der",f("der")(c,m=2,scl=1.25))
        record(tag+"int",f("int")(c,m=2,k=[2.,-1.],lbnd=.3,scl=.7))
        record(tag+"int_zeros",f("int")([1.,2.,0.]))
        record(tag+"der_zeros",f("der")([1.,2.,0.]))
        record(tag+"val",f("val")(x,c))
        cc=np.stack([c,c*2],axis=1);record(tag+"val_series",f("val")(x,cc))
        record(tag+"val_complex",f("val")(np.array([1+2j,-.3+.5j]),c.astype(np.complex128)))
        record(tag+"val_float32",f("val")(x.astype(np.float32),c.astype(np.float32)))
        record(tag+"vander",f("vander")(x,3))
        record(tag+"val2d",f("val2d")(x,y,coefND))
        record(tag+"val3d",f("val3d")(x,y,z,coef3))
        record(tag+"valnd",f("valnd")((x,y,z),coef3))
        record(tag+"grid2d",f("grid2d")(gx,gy,coefND))
        record(tag+"grid3d",f("grid3d")(gx,gy,gz,coef3))
        record(tag+"vander2d",f("vander2d")(x,y,[2,1]))
        record(tag+"vander3d",f("vander3d")(x,y,z,[1,2,1]))
        fitY=f("val")(fitX,c)+.02*np.sin(11*fitX)
        record(tag+"fit_y",fitY)
        fit,info=f("fit")(fitX,fitY,3,w=weights,full=True)
        record(tag+"fit",fit);record(tag+"fit_residual",info[0]);record(tag+"fit_rank",np.array(info[1],dtype=np.int64));record(tag+"fit_singular",info[2]);record(tag+"fit_rcond",np.array(info[3]))
        record(tag+"fit_selected",f("fit")(fitX,fitY,[0,2,3]))
        cls=getattr(module,clsname)
        s=cls(c,domain=[1,5],window=[-1,2])
        sx=np.array([1.,2.,3.,4.,5.])
        record(tag+"series_eval",s(sx))
        record(tag+"series_derivative",s.deriv(2)(sx))
        record(tag+"series_integral",s.integ(2,k=[2.,-1.],lbnd=2.)(sx))
        record(tag+"series_roots",cls(rcoef,domain=[1,5],window=[-1,2]).roots())
        if name!="polynomial":
            record(tag+"to_power",getattr(module,prefix+"2poly")(c))
            record(tag+"from_power",getattr(module,"poly2"+prefix)(c))
            nodes,w=f("gauss")(5);record(tag+"gauss_nodes",nodes);record(tag+"gauss_weights",w)
            record(tag+"weight",f("weight")(gx))
        else:
            record(tag+"valfromroots",f("valfromroots")(x,roots))
    cheb=np.polynomial.chebyshev
    record("poly_chebyshev_pts1",cheb.chebpts1(7));record("poly_chebyshev_pts2",cheb.chebpts2(7))
    record("poly_chebyshev_interpolate",cheb.chebinterpolate(np.exp,5))
    fails("poly_bad_division",lambda:np.polynomial.polynomial.polydiv(c,[0]))
    fails("poly_bad_derivative",lambda:np.polynomial.polynomial.polyder(c,m=-1))
    fails("poly_bad_power",lambda:np.polynomial.polynomial.polypow(c,-1))
    fails("poly_bad_degree",lambda:np.polynomial.chebyshev.chebvander(x,-1))
    for name,fn in [("add",np.polyadd),("sub",np.polysub),("mul",np.polymul)]:
        record("poly_legacy_"+name,fn(c,d))
    q,r=np.polydiv(c,d);record("poly_legacy_div_q",q);record("poly_legacy_div_r",r)
    record("poly_legacy_der",np.polyder(c,2));record("poly_legacy_int",np.polyint(c,2,k=[2.,-1.]))
    record("poly_legacy_val",np.polyval(c,x));record("poly_legacy_roots",np.sort_complex(np.roots(c)))
    record("poly_legacy_poly",np.poly(roots))
    record("poly_legacy_fit",np.polyfit(fitX,np.polyval(c,fitX),3,w=weights))
    record("poly_legacy_series",np.poly1d(c)(x))
