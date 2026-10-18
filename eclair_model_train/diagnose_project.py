#!/usr/bin/env python3
"""One entry point for the ECLAIR/DALES evidence audit and bounded diagnostics.

Uses the project's actual loaders, voxelizer, model builder and canonical
evaluator. Never trains a model, changes a checkpoint, or changes cache payloads.
The compressed frozen evidence below is embedded at packaging time.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tarfile
import time
import traceback
import zlib

import numpy as np

FROZEN_B85 = "c-rk<*^V1oa{ZM7pR=XAFM1n$4A_7NelXydhLFg$iY;!#B-L7=fA=}}X0phtQmaZ^m(<y6P-HH5i8v8)B7#i4U+j;YVe#VE_k6Q>u~1P9w<s3t?r^<rap0c<pN8#m-*ouYHLGg5$C2i`YhLZP%gq5tpAXHxIyBeM*Ge}-Wu<uDiE`f!$JJrr+k6|2?DpF-P4WC@fZ=)F4TopD1Bcyi`sLN)#rs8D9V-4^9n;5dQ`M^u3p(tNU9tFWc{p_Yaya0KcqU}A*jJl&yDt0Ha#y~2J6$&(-ER-op)1=CWBrQj3+?dw5*ur(RafGwt1g|1wvf++efgnS)Z9@yEZ^ab!pbVF7z1rTw&A?RbQw0n<qzHATx63F#vz)ayWXz5vmsW?w%V87n_ahGt~&~59A$WZ8l>;4!x56DOqQEwb!@w`*>3kOj`Y=Pz@8z}ZZj+oZ_9T1dO2+Oix<vXZL#gezQf&{q1<jT`JsHZcwvjhap=m^O{ZaabaPzS5c9S##~UZP@?<_VUUPBcsvbvy7_atKTUJeTTti#kZb-shRdu&2hh0Tby`Lm=+_YW4+;nZZq*}3)tIc-LwYA-w#fx90z%fi$;__jj45a*!kNm#fzv}ilVv5C%n&GfXE+vKlEUM-}K}afpv94Zq<!TGvpJbO;|9bm|>q<Jul@_DJEV5F|-M(wK>)r7%ZG95s`QEes&xc#Ss}9TUh6|$d%H4Lmg8tWCwJEFBYB5P;I$r&@-BS-I@D3d$vhDi>C-LFKgd~m#Gp)*gxq^ADX`gAR;n{BkCM*^Q*cI0LdfEMsg&n#pj7+ipx~O)r#MiJV*uY_5EjORIjL7+|YYzF2*AUz%hUF@E+vb`|qS`v#!v)vf>kD%p!PdR*_HWDS4q$SHBP+b!99JtU>`j^MvCH@ws@DML?%gEl^m(_r%AYa!dfQSQ*+CVH-vOcR@5!AcB({gQJD%@X+v=bduBfiA_TAN}sjQsU_G)<;l2giBPov@RaKa0b0&i2UvF3!V;jmw}=ZC1Sd<W)U?XIh7OIOu;jSq@<t+ubL;a!64gzN>-Hgsw6w_M-8TJOpm*<y3meto=ks9RoL6D`jO;Sn1>0lyf9la%p~#S4wCAK#SMUDXa}`oUl2a($Idftpo<-m+abU3uRU(}`gX&j!Pj+|q3YUoYFX+rT41F3#LF?ORrWIXLcwf?pRk%=J|u__8$9l#5%)0B(Rk;&hv>BcovGRyc%<PbT&z0eYa4AfjUNrsO#5gjKE{?_m<@EIz$WQDu_<+gsPZy}0(>bXJM!b3Xe^x*XDnOD^xg{XQM$L)*>jZJD%^r$1Ni2Ri$*xvn;wBqrXrJs!@F&_Ql@2a5guVzp1JWVjqLQMJ{2F%63N*i>)xB|>K3mvFm2ZN7!oCnC@s+iH=``qpk>p0B!YS5k%Lt9-Z|Hq~x`s~i&IU|Hi$V@R;#w4xif(C)Xp|61f7wRCr%!!s76bGWreh@5T}0@B)lAP>0ar}vXtCRB_o;u<*&{z~?6rlk*;Ztg>IMx&hhIE`w-I)+EHsB~AE-F2Ga$0AX3B2lk8XyOg_k=32Im9$;HR%H%hVR5<|J^~l+tKS!&W?^Zp{Cc8d`JUleXAqrDmv|aKW#6-|cI5^k0B|{Mvz#e4F;NehNQp{2(Fy1#!P&y46$`g`;fsY&<{UsoK1_q>DEyEmKj)>VMU8R!VK=75p3}|=V&9iAeuDbxJJc^llaiK}-2Byk3l@lFzwWL;ZBjU<2KU?w(Ad1<oBFt1wad-bkbHXH=WhEuu)G3&%)8z+c=PfB95Q@U3aLIYCUt#BJD7$r=@@)eb0dNZ{=Q4WBaLvMcE)hFfiA|#H(i=3#hxSN(UxtB+J!D`;R^pUYv($<d9ahF+TG}B-~H>j+~4A^1H3+@nfyirV@Yu=yP6c-AcJ)Hi~>oOM8m^Fi6o|69=dh5`_rKVxN3Tn+$Qe<H>e<EZf6=^3y~(cIhJn8^KJXXIV~Kn_sh+z31IGAE;l!E2AJpBu(z+s!#X=&!LWL3cOj`gt03)XF-aU82*cFXssiW`s}EBEIX``_m!y@K_npsyI2o%Q$szoTPQ>B-YG9}4Brfq7stP_Uv+xw#vQYLkAYK+o0|)5y06A}#Jw4J#Lu&RY({<eD-NXj4@Ap76>=SPg97b;`p2=^N_JE||{AnE%c25!g&rdQAIo95uHHA$`VmOPOkuNgvRr#|ZN^r>Z@%0va{diLH;m^uxl)X2n!LF<>(hJt_@#3aWZaO09hQ3qU<G)VYBY)zwM_r^n7g^6Q@{jYLi?nCF_}%5{cNeF>ciIzvz^vz-22bh3BT}C`$-!r5KH<}ukGjZw)OqHkgLLX5^I5*p|M=1x@>)*0-5SHSWtwHKA<s4BuT(=$s7jDoR=8m}ESu9h+bu>&MOMtdXT1{}TpU4<m=?6lzVG&sbeRF3F1#DQu8t(RHK24N?Eu_J$g34rNo`@=e+irOiqXsG*(*SGy`v)u`7i=Ix!2|KRK`umFNu0Sv0Fh@>$FnFbH;b76A9F^c{`HY>9R!%DRosZS4p2TEB<B(aM9{S0aG}!OxpI#{w<j^sBEePZxSVLwyV{0AO|{er|(s?zr)5JQqDg^?_AHH>-lp%f3D~Mr1gBVrt*eHr4rLBt?{GW_mbJ(bH^XB<2$r?MAIKqXYmp0oWkId7~Gd+3%U9<3gQ-Fd<2)imumh=Z3ND(|By_7MkfIq%*n#d{`A~Qm^%q`C*dn}66QqWr%5E{*3I0ynOiq=>t=4<%&nWbb@PL^Zhqzj_cDu9B0et5y^{oHh#l8miVI(!M{>V8kNi^_QPF#Go;&}~zmqq9VI!)&nyLfahJhi?GB={;M%2@9M2(`D6Ugt{iMlUAyEs^~=aU`8cy?}H&CRR7PV;IE&+n%*DXz|Kt)Hl^RU|!CtKqJ(J-FqSKtb5<b_f#5G|IbifmbaNnKuk05Nw42kajjjx-n**t&Waq4U&?W$T-ZMZGN}|{B<?Fx`;Y}21dak^<=j^2TzcL0<X`Z?2tUyMNugg`ik)K{5)5EKEA(wa@vy4Jo@C6BQz-HB<f>HlswC!ZjpH@Qg~xN@`z08-i+Y>oJW4^c471`{$a}V8SO&sFQ<8@?YTI&3+Hy>+%A0D?ZP?ToBM@xzi{ps&i%r<U-*~n7k<s6!rV!Gik-wSD=LJ)LQP?=Cw%{U!da^P+2w>g>GtPV6TY~C8pC-t;ott_|9<)VfB)~-YoU%7T&9`kChFWootvn0Ry{XS=O*ghME$}h>coxbHtJ8>Mg_^ncK79NN#BCjNDj&*TH+!L_U6Qekgls$&jtx20>(?Fiz%p$(*4)5+8lCB98V00H(mDn%gyexb*sXSt<(U`MUIqUbaCWP8g*aN1g!6uO?kM6O0T!8cDfgWA6L}1S6z8k?M5}pd=mfHtnm*QaW2LLqb!IlMwu&Z+to2;gcE8G)&8nG$>DtS$e%!LmkH943tR<e$Yxk?-;8y<<Z{ok_2}03tt5BO$R5?<opOz9Q?6HE*5|!QTmH08Z}gwY&Az19d%i_KcYEh<@7(R3yS;N>I(K{LZtvXfox8nrxA#eRd%vvuG533)e7|?De*6WhA7L(je5H2u*QkBmQy2KIZRwX;rpFv|&;G@-hd(1LWB9x>0s#H`t0*r1yex3U1!XK=3S+!*v5?*wC2<=3F2+I$CDV1%O0TUgbaWyZfv^~&GbUZ+r1eGc5y-R#gQOdbR|-T<du^mC@YZVwlQRy4fl<{6<uJS!c^%TY!U>B*QuD=WZweEnER3*tWAattNKqRs(_yVz8C`g9F|U_6W^}qJ7>|gpC3hut2ojq#92bjn)@dBI(|AUCTi6&eJ9i$iaqU$Rv@IluP~zoF`jzc5lTe3kd9Q>qSOLQ9CZz+Zf;n}g*41u@tfNE*c1=S{#iOyJ0QSsM2&l^%E+=W{+1qMOcQO`AK63YTc&ubhL*$zu{jae+1|QygX{;$i)ZPbZ%vfj%Dy8m0igX-BsU#E-0Fa;spiwAitfv?yl;kXQWQ<dI70TwJmD1FSbl!O&L__BeHwa0XQFqqJ0vkFh-^apg=u-Nqp->r>DS$^PSmAUAYEn98!b-u}OqyM3tk-EtRHTal89)%qgl7RXZNS|EXb;4iatgqfHjc=|TCh)PU;wy6aekAKC}9+M<Ge8b0wxkpLko)qTF5UxIhcHgC%O6l%SWN2Y=FB68Wt*Igdthvq6LvrzzV6pbeWFysg$)B77BzyH7yPvNCaro7g|oD2cjTh&(sLI0-87rd=-X($&c!RCIvhb(vU(%Z80{S5Qb54T$qHv@j?>3SVwXi04H9?^Onvd0OJ^n3w91zaNbTX0#j)$TaLRmR>gC5TL2sG%SFC;f39t2vOk_zc#y7z(;?CnA$wY_r?TM~Bm_*CnlED=gt~@C75pVUM!-?ZTWAQ@8rr)ES=8_h)FquS2ggN3PqJH>e!3*m4Z!sW`XF2g0ULO4j;SYY8;_s>udm>W3Im-w;b`FFc>~NTI4P_%WOWOKGp#ZdFsy<X!`SJZ<n<w3s2g4ki7&uz6?7h-p1vRPWcmk30!SHA6@fHHhcE#9hFL2(K$DCT_TuPLjiIl00Ra+HhAlEM!>wx@82pu@uC##U5k?JVZOJ8UOg>(@pdoy3G-wBczNg^<Eil^v6loYHZec2p7*7KaL`8tx&=?WdhlDK<4+l*kevr@Dw0>)eOdv~xxLD2zXTggZWH1y^njq$*u^G1j0VWJ|M_Q8EC?0p1$_K=&<f7oeEVm8t<AnDD5fKanm}5ZVKn9a3(2k(aF@(tD5x<;`AS9$kUlt*j?c?w>w)g(XVLa&H#(gk)=y)yB1Ir;NfW<?V8axcL*MyIDw0$^oT&)pFXrE4j5h8dXm?}DK<8_SGVn|L#f|EcMj1r2I$N+n^R1`u?VF4x34W-~a_!+tfrXr&StT#C!#v}M4u!O+dVEf}897ZaWC`d-Xhbw_p5CtrNCk3Dj6VOC)a)NL<A>jd7CG*jYC?E`oRoJX4KrygUllKhI7C0(aM2@NiU7Q7|;B7#K5To!}5#W{+1&bYAphv-~ybJ*wL&S+cI(j}CZ!(><(F*~cgdhMOSPq<aK|#Z1gp4v+l7j|3*+7uiDyopmMoe%_E*uG^OL}KA1BTCR1Q8w~%A|P1OJTW8C7`ZAMgorn_!tAM(XHh*Aq<hT7F0M0s9^pPGe_W36w%4G!DEnU;E#t_&S>Jm@F0$HEmI0GQX?-@4Q+vT=vTqvYWJm4=4&hYb7IUV^{i6Va4<2M*c$eZ!~^`ESs<uCg1t|^92NzaYAp>A^f4N;d0HN>g~35>3VLHC0$@~xz!<1Kw1khv$PoewBytTn77`Mw7)&0fIxWY=6c-F)g~%C+^-0^wcs0Wz;@+r2ObSkrPjN0)Wd6cG7Rl^|awA4k$fB`d^57?@+(&%r;&HVGSTd}_DkMi>e+ULTp&wZ_?a+?|4v7E*Er1^*%v3U^q_MzxTLx74Z+HX%KxbCyqd@Y`%tBe#8kA<;0wjg5nnuWk4pBjjl-?n#1j-#LpiUJBzzZavix7M~Manjl?_f!UF5i&2c<>i$leoxh^};b3jtWSKPBKqq>_C_RYz$c>Cm$1)<b@n}A-!=VpR9EtYT+(~uHY^(0$6KiyB;I~De%)vB0mQt4@xbdbp*X4!X<-Tfh*DH!af5N|1pO&f|`a7J)@?IHZn`FY)O&KJC+=>++q2^QmO-iNnorb_{fUTl;Ndx9;d09Xc@1O?Iq7KtqJZ3K83sy9zrA2v{A{^l7Uouid4(UDFvcAYiCnVK_A9|J|>Xv($`yJ(Z$I0S&<7oy^P{9zmR#bN3bwvdIcS{Wb!UxzlID6Z;DhWWX%XR#S9GYm~_d~Prx9+reGMJ_6t{{$pi=i<={!c2W{g2h`u?h01LRxGIKFZT^PEh5vkN@;XDdNSY{RU7%F>}7{+iSSflc+6KN4rp)#Ch74%a?3edq~n9tq^i_v@!w2fH>Qx{N97L1Uqkk!JMzy&gIMGyct1cb3`mslkHqfZ4%W))a2hLKD<q)luGIZX0r$Z{^T3YY^#307@Q_+ppwEuzZfLG)?TAZTmMuu?rDdduP@nIe=HNUZ6y6NzWI_As?X8?++(lZsVtn5a#e1PP-|!~q^0`i8m)C2t+Zjzumv=~P8b_HT4bf3+bMu(_bXyHj-nq%Nj<5f#V$W?V34@`yF8iJF+HVT_p?>cE&%sBwSDCb7tiun-VM5#A6NDbDx=slx-G%=0MXfWYpkR``RMI6kRX!JMDz{#d+097d4!sYd}HYw44<q+>DkC!WP6sYpD8lR3&mtt{&z@AD>&WTQo#Nio?lHDjGYIxz8(Ij{uHDk35g4H_;9ij<Lw$f*RKLXl@?$6AF3i~t~9OP(@<hiozA<OEnrRT3n><RWavc)CEgNJc7R1d<(xRkKjxgDr9=M3Mjlt{Uv&k~##~0N4@!a{39jDLw-v534`>2x%bkLO`%2`P31F3Tg%hH<tep-dU4DNCpkpw8W$-<m4<GsQ|SHJ2FwB*=7P-B*`V9!0^u6Asof2w9M)OJ0z_3!vknc9~isFa-^42bOE>2ioL3oIu>IqLZy-eUVKg%T&hQn1yq))*rGCH^_e&`m*vR7eZYHRddm{UQzQ=$eLnGneS)SFgkM1H_ljlVR4{^#CBL8&RcEcxkpo)im~JTADGLDJDI@9fg-V>kvq5Q-Rzbo<$_kQ)7pW!4Q0kHT!>72E^F#SQwIInnlP?IKdP}IGPiBXbv6z7&xhK{*HF?A6=HbILA;c-3V-1mPpFEY(DftTW2Adq57OX{qudqbr*+NC?gry)RX~SA2ofRWoFa#scv2n)q323uya>Z2fRLoG6VSnp_H(2IV7Cr+e<6U@qyy4OKc>rr*O)yngsCe+W$Q}Xo%o;KCy_lOAhK$l9{!rP-Hb4iddj<zq0N6ODdQ(qRO4>jB;B0evq;9+_0TRr^+FWl?OaMmKU|27<6Kfy}*qr29@<(KztZa-$1+WG%21l1*M%G|QYtWpLhjaP?(xZ*NqzwiE6h2Hx6<H^L8nodhzgW9GEcqqm@%!G_)o`7^T|exb=jq!e`TNh~jk~w$R}LK_Y$Ll$RIc@%>}`W%FgekMs_pvdTHQs}sJ^XgBO2B9qOD}x2N1*15Ih(?eVZkZXnoyE)3mCUw(6Q%)om^Kr3DzciB(m_9yw>FZ5LuAgg2r!y~6~ZY&$Ubam4d4j^=@dsFmu1<n)!SWiLV-Le!?MF^%(0Y<pSj*jnq`%ElTv3C`(`^NCiC15ZCY7*`u&-F9uKtI+kG?p<qqD;g2H4wfu!>zZ0f(KXgpy+A-gMo{%0NlN1z6Q<Sj*A1g;^0<9%Jw(;nR=cik8y`AhTB)lxnhNet*EJ?K0ejK0f&qY1S9Dpv@v-&I`M5cdq+yYDTVboFHMQvbrfGa<uo8pgh=rhRwX2+}0sIbWy9!m^^``G!Wjbt3R^zbgJ5qT>)l_{1zrki&i{-QiF&gn3$z#)0QsHdh_W~)k^<8YcS_^&$p~JCCSJsRpPT%s$eq$V4+cp45;7`IsH4!&gt-_$ucCCZofGlh$B8*A4S~U>6t6Lv>-}hB3`def3TbLtiD-ZRvQRWn2S=X|YwyV0f?|~2;Z<Y2?psKZnGyuT5QhnVx*+UYw9S5$CSNzF99@8NUjvbslm}?KYRVp^I6Lkw5E1<YjZPc}{+0&59=sLQ_+a4+v!h~)@Li)kR2w2ww<`x)L4wr-VU~8diWCd{9D(cFCU&AZ*p?0oqZQEc<)j1PuW29b;-eg27L$7Lj*SQ$`U>bvP)obkE_DzIgLgyG1!&L^`wKcR4sYn^CR`){jtF70|;jrB&CxM7?{*@^39c%B61gBu}pQoGuCZ3t~-}ois`Dq*T(>6ZiX&c|}H#X+O10UCLv-kOF4fE3)=6|nwej>w9`tLR0`6Bwqzh?fiZ<6bee3N`!Q75LXWcuy$L%$?`b7%SqWO-sC-}WC)e#Fn)?_J?Xe%*fIJ<q5Xol%83wx^JrY5n*E-KHDz@8<p~xSdlhtidDrtKqNqOr)D7eWg%~|88mv!U*$70-20@=8rRf{EP5MgqfK>&h&Ank28Io>Ekz`k0%cDWa(pC;Y=S#`gk{7o-g_<Ju2dI*<U3d-=e+lzJB|CtXCTG)XS#xqp;>jVLj1DVa*R_`YI1*GV_C(<_9y)4`!Ml%=9!L%rrlI>F0R((gPpX^MfU-=QMbt{*TC3&pG2MbNH-e^}!Fsd0eu3XMvx{Rn-^es^|3OB3+%IG5HfvnOpUOnV$aeI>mo4Km0$-%mvf"
VERSION = "1.0.0"
NAMES = ["non_common", "ground", "vegetation", "building", "wire", "pole", "fence", "vehicle"]
EXPECTED_TILES = {"eclair": {"train": 1059, "val": 62, "test": 125},
                  "dales": {"train": 26, "val": 3, "test": 11}}


def frozen():
    return json.loads(zlib.decompress(base64.b85decode(FROZEN_B85)).decode())


def clean_json(x):
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    return x


def write_json(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean_json(x), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_obj(x):
    return hashlib.sha256(json.dumps(clean_json(x), sort_keys=True).encode()).hexdigest()


def payload_digest(raw):
    h = hashlib.sha256()
    for key in ("xyz", "native_labels", "return_number", "number_of_returns"):
        if raw.get(key) is not None:
            arr = np.ascontiguousarray(raw[key])
            h.update(key.encode())
            h.update(str((arr.shape, str(arr.dtype))).encode())
            h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def stable_seed(*parts):
    return int(digest_obj(parts)[:8], 16)


def file_identity(path):
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p.resolve()), "exists": True, "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def cm_from(gt, pred, columns=8):
    gt, pred = np.asarray(gt, np.int64), np.asarray(pred, np.int64)
    valid = gt != 0
    if gt.shape != pred.shape or np.any((gt < 0) | (gt > 7)):
        raise ValueError("Invalid common ground truth or incompatible prediction shape")
    if np.any((pred < 0) | (pred >= columns)):
        raise ValueError("Prediction outside declared label space")
    return np.bincount(gt[valid] * columns + pred[valid], minlength=8 * columns).reshape(8, columns)


def metrics(cm):
    cm = np.asarray(cm, np.int64)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = support + predicted - tp
    iou = np.divide(tp, denom, out=np.zeros(8), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(8), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(8), where=predicted > 0)
    supported = support[1:] > 0
    return {"cm": cm, "n_valid": int(support.sum()), "support": support,
            "iou": iou, "precision": precision, "recall": recall,
            "legacy_miou_all_seven": float(iou[1:].mean()),
            "supported_gt_class_miou": float(iou[1:][supported].mean()) if supported.any() else None,
            "absent_gt_classes": np.where(support == 0)[0].tolist(),
            "non_common_prediction_fraction_by_gt": np.divide(cm[:, 0], support,
                out=np.zeros(8), where=support > 0)}


def grouped_majority(inv, labels, classes, ignore=-100):
    """Sparse counts, deterministic lowest-ID tie break; no Nv x C allocation."""
    inv, labels = np.asarray(inv, np.int64), np.asarray(labels, np.int64)
    nv = int(inv.max()) + 1 if len(inv) else 0
    valid = labels != ignore
    if np.any((labels[valid] < 0) | (labels[valid] >= classes)):
        raise ValueError("Out-of-range label in majority diagnostic")
    pair, cnt = np.unique(inv[valid] * classes + labels[valid], return_counts=True)
    vi, cl = pair // classes, pair % classes
    maximum = np.zeros(nv, np.int64)
    np.maximum.at(maximum, vi, cnt)
    winner = np.full(nv, classes, np.int64)
    at_max = cnt == maximum[vi]
    np.minimum.at(winner, vi[at_max], cl[at_max])
    winner[winner == classes] = ignore
    return winner, vi, cl, cnt


def quantize_like_evaluator(xyz, edge_m, norm=10.0):
    # Preserve the actual float32 divide-then-floor order. xyz must already be local.
    return np.floor((np.asarray(xyz, np.float32) / norm) / (edge_m / norm)).astype(np.int32)


def voxel_diagnostic(xyz, common, train, train_to_common, edge_m, norm=10.0):
    q = quantize_like_evaluator(xyz, edge_m, norm)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    majority, vi, cl, cnt = grouped_majority(inv, common, 8, ignore=0)
    native_majority, _, _, _ = grouped_majority(inv, train, len(train_to_common))
    projected = np.zeros(len(majority), np.int64)
    ok = native_majority >= 0
    projected[ok] = np.asarray(train_to_common)[native_majority[ok]]
    valid = common != 0
    occupied = np.bincount(inv, minlength=len(majority))
    valid_count = np.bincount(inv[valid], minlength=len(majority))
    distinct = np.bincount(vi, minlength=len(majority))
    mixed = distinct > 1
    class_points = np.bincount(common, minlength=8)
    contains = np.bincount(cl, minlength=8)
    survives_common = np.bincount(majority[majority > 0], minlength=8)
    survives_native = np.bincount(projected[projected > 0], minlength=8)
    correctly_constant = int(cnt[cl == majority[vi]].sum())
    mixed_points = np.bincount(common[valid & mixed[inv]], minlength=8)
    return {"edge_m": edge_m, "n_points": len(common), "n_voxels": len(majority),
            "valid_common_points": int(valid.sum()), "class_point_counts": class_points,
            "class_containing_voxels": contains, "class_common_majority_voxels": survives_common,
            "class_native_majority_then_map_voxels": survives_native,
            "class_points_in_mixed_voxels": mixed_points,
            "mixed_valid_voxels": int(mixed.sum()),
            "occupied_voxel_point_count_quantiles": np.quantile(occupied, [0, .25, .5, .75, .95, 1]),
            "constant_voxel_accuracy_ceiling": correctly_constant / max(1, int(valid.sum())),
            "common_majority_backprojection": metrics(cm_from(common, np.maximum(majority, 0)[inv])),
            "native_majority_then_map_backprojection": metrics(cm_from(common, projected[inv])),
            "warning": "Accuracy ceiling is not an mIoU ceiling. Majority mIoU is diagnostic only."}


def remap_paths(obj, maps):
    if isinstance(obj, dict):
        return {k: remap_paths(v, maps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_paths(v, maps) for v in obj]
    if isinstance(obj, str):
        for old, new in sorted(maps, key=lambda x: -len(x[0])):
            if obj == old or obj.startswith(old.rstrip("/") + "/"):
                return new.rstrip("/") + obj[len(old):]
    return obj


def offline_audit(evidence):
    rows = []
    for item in evidence["evaluations"]:
        m = metrics(item["cm"])
        rows.append({"run": item["run"], "method": item["method"], "source": item["source"],
                     "target": item["target"], "miou": m["legacy_miou_all_seven"],
                     "reported_miou": item["miou"],
                     "absolute_error": abs(m["legacy_miou_all_seven"] - item["miou"]),
                     "metrics": m})
    priors = {}
    for domain in ("eclair", "dales"):
        row = next(x for x in rows if x["method"] == "M0" and x["target"] == domain)
        support = row["metrics"]["support"]
        priors[domain] = {"support": support, "point_fractions": support/max(1, support.sum()),
                          "scope": "Known test-set composition, not source training exposure or a pure label-shift proof."}
    return {"canonical_evaluations": len(rows), "all_miou_match": all(x["absolute_error"] < 1e-12 for x in rows),
            "known_test_class_composition": priors,
            "rows": rows, "limits": ["No historical checkpoint hash exists in the supplied evidence.",
            "No seed significance or causal root cause follows from confusion matrices.",
            "Eight checkpoints share the core backbone; they are not eight architecture families."]}


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.evidence = frozen()
        self.events, self.failures, self.dataset_rows = [], [], []
        self.workflow_completed = False
        self.interrupted = False
        write_json(self.out / "STATUS.json", {"status": "running", "version": VERSION})
        self.log = logging.getLogger("project_diagnosis")
        self.log.setLevel(logging.INFO)
        for old_handler in self.log.handlers:
            old_handler.close()
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(self.out / "run.log")):
            handler.setFormatter(fmt)
            self.log.addHandler(handler)
        self.log.propagate = False
        self.spec = {"version": VERSION, "arguments": vars(args), "python": sys.version,
                     "platform": platform.platform(), "script_sha256": digest_file(__file__),
                     "frozen_evidence_sha256": digest_obj(self.evidence), "common_class_names": NAMES}

    def task(self, name, fn, resume=True):
        p = self.out / "stages" / (name + ".json")
        if resume and p.is_file():
            obj = json.loads(p.read_text())
            artifact = obj.get("result", {}) if isinstance(obj.get("result"), dict) else {}
            artifact_ok = not artifact.get("artifact_path") or (Path(artifact["artifact_path"]).is_file()
                and digest_file(artifact["artifact_path"]) == artifact.get("artifact_sha256"))
            if obj.get("signature") == self.signature and obj.get("status") == "complete" and artifact_ok:
                self.log.info("REUSE %s", name)
                self.events.append({"stage": name, "status": "complete", "resumed": True})
                return obj["result"]
        start = time.time()
        self.log.info("START %s", name)
        try:
            result = clean_json(fn())
            record = {"stage": name, "status": "complete", "seconds": time.time()-start,
                      "signature": self.signature, "result": result}
            write_json(p, record)
            self.events.append({k: v for k, v in record.items() if k != "result"})
            self.log.info("DONE %s %.1fs", name, record["seconds"])
            return result
        except Exception as exc:
            record = {"stage": name, "status": "failed", "seconds": time.time()-start,
                      "signature": self.signature, "error": str(exc), "traceback": traceback.format_exc()}
            write_json(p, record)
            self.events.append({k: v for k, v in record.items() if k != "traceback"})
            self.failures.append(record)
            self.log.exception("FAILED %s; continuing independent work", name)
            return None

    def prepare(self):
        self.repo = Path(self.args.repo).resolve()
        if not (self.repo / "src" / "eval_cross_common.py").is_file():
            raise RuntimeError("--repo must name the project checkout containing src/eval_cross_common.py")
        sys.path.insert(0, str(self.repo))
        self.ev = importlib.import_module("src.eval_cross_common")
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available():
            raise RuntimeError("A CUDA compute-node environment is required for the complete diagnostic")
        self.device = self.torch.device("cuda")
        self.torch.cuda.set_device(0)
        self.ev.set_all_seeds(self.args.seed)
        self.spec["runtime"] = {"torch": self.torch.__version__, "cuda": self.torch.version.cuda,
            "gpu": self.torch.cuda.get_device_name(0), "numpy": np.__version__,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": self.torch.cuda.device_count()}
        if Path(self.ev.__file__).resolve() != self.repo / "src" / "eval_cross_common.py":
            raise RuntimeError("Imported evaluator comes from a different checkout")
        source_hashes = {str(p.relative_to(self.repo)): digest_file(p)
                         for p in sorted((self.repo / "src").rglob("*.py"))}
        source_hashes["train.py"] = digest_file(self.repo / "train.py")
        self.spec["source_hashes"] = source_hashes
        critical = self.evidence["critical_source_hashes"]
        drift = {k: {"expected": v, "actual": source_hashes.get(k)}
                 for k, v in critical.items() if source_hashes.get(k) != v}
        self.spec["code_drift"] = drift
        self.preflight_errors = []
        try:
            importlib.import_module("scipy.spatial")
        except ImportError:
            self.preflight_errors.append("scipy.spatial is required for the bounded geometry block")
        if drift and not self.args.allow_code_drift:
            self.preflight_errors.append("Critical source differs from the reviewed snapshot; see code_drift")
        self.runs = []
        mappings = [tuple(v.split("=", 1)) for v in self.args.path_map]
        mappings.append(("/scratch/m23csa510/e0_results", str(Path(self.args.results_root))))
        overrides = json.loads(Path(self.args.overrides).read_text()) if self.args.overrides else {}
        self.spec["unavailable_historical_comparators"] = []
        for entry in self.evidence["runs"] + self.evidence.get("historical_runs", []):
            r = copy.deepcopy(entry)
            r["config"] = remap_paths(r["config"], mappings)
            if r["run"] in overrides.get("runs", {}):
                r.update(overrides["runs"][r["run"]])
            r["checkpoint"] = remap_paths(r["checkpoint"], mappings)
            cp = Path(r["checkpoint"])
            if r.get("historical_comparator") and not cp.is_file():
                self.spec["unavailable_historical_comparators"].append({"run": r["run"],
                    "path": str(cp), "reason": "Existing historical weights not accessible; not replaced by an untrained model."})
                continue
            config_file = cp.parent.parent / "config_resolved.json"
            if config_file.is_file():
                actual = remap_paths(self.ev.load_config_any(config_file), mappings)
                for field in ("data", "model", "epochs", "loss", "ocons", "method", "eclair", "optim", "sched"):
                    if actual.get(field) != r["config"].get(field):
                        self.preflight_errors.append(f"Run {r['run']}: {field} differs from frozen config")
                r["configuration_path"] = str(config_file)
            r["checkpoint_identity"] = file_identity(cp)
            if cp.is_file():
                r["checkpoint_identity"]["sha256"] = digest_file(cp)
            else:
                self.preflight_errors.append(f"Missing checkpoint: {cp}")
            self.runs.append(r)
        self.spec["runs"] = self.runs
        self.ds = {}
        self.luts = {}
        for domain, run_id in (("eclair", "29306"), ("dales", "29364")):
            r = next(v for v in self.runs if v["run"] == run_id)
            cfg = copy.deepcopy(r["config"])
            cfg = overrides.get("datasets", {}).get(domain, cfg)
            cfg.setdefault("eclair", {}).setdefault("train_review_categories", None)
            self.luts[domain] = self.load_maps(domain, cfg)
            self.ds[domain] = {}
            for split in ("train", "val", "test"):
                try:
                    manifest = self.ev.get_dales_manifest_for_split(cfg, split) if domain == "dales" else None
                    if domain == "dales" and not manifest:
                        raise RuntimeError("Exact DALES split manifests are required; no directory-inferred fallback")
                    ds = self.ev.build_target_dataset_one_split(target_domain=domain, target_cfg=cfg,
                        source_cfg=r["config"], split=split, target_manifest=manifest,
                        force_raw_cache=True, logger=self.log)
                    # get_raw is read-only; fail rather than silently use missing ECLAIR caches when required.
                    if hasattr(ds, "require_cache"):
                        ds.require_cache = bool(cfg["data"].get("require_cache", False))
                        if hasattr(ds, "require_cache_metadata"):
                            ds.require_cache_metadata = bool(cfg["data"].get("require_cache_metadata", False))
                    if len(ds) != EXPECTED_TILES[domain][split]:
                        raise RuntimeError(f"Split count {len(ds)} != canonical {EXPECTED_TILES[domain][split]}")
                    self.ds[domain][split] = ds
                    identities = self.tile_identities(ds, domain)
                    absent = [x for x in identities if not x["raw_file"]["exists"]]
                    if absent:
                        self.preflight_errors.append(f"{domain}/{split}: {len(absent)} raw source paths absent; stat-based cache keys require source files")
                    missing_caches = [x for x in identities if x.get("cache") and not x["cache"]["exists"]]
                    if cfg["data"].get("require_cache", False) and missing_caches:
                        self.preflight_errors.append(f"{domain}/{split}: {len(missing_caches)} required caches missing")
                    self.dataset_rows.append({"domain": domain, "split": split, "tiles": identities,
                        "manifest": file_identity(manifest) if manifest else None,
                        "manifest_sha256": digest_file(manifest) if manifest else None,
                        "eclair_metadata_sha256": digest_file(Path(cfg["data"]["eclair_root"]) /
                            cfg.get("eclair", {}).get("meta_filename", "labels.json")) if domain == "eclair" else None,
                        "config": cfg})
                except Exception as exc:
                    self.preflight_errors.append(f"{domain}/{split}: {exc}")
        for domain in self.ds:
            members = {s: {x["raw_file"]["path"] for x in d["tiles"]}
                       for d in self.dataset_rows if d["domain"] == domain for s in [d["split"]]}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if members.get(a, set()) & members.get(b, set()):
                    self.preflight_errors.append(f"{domain}: overlapping {a}/{b} file membership")
        self.spec["datasets"] = self.dataset_rows
        # Output path does not affect scientific identity; all source/cache stat identities do.
        sig_spec = copy.deepcopy(self.spec)
        sig_spec["arguments"].pop("out", None)
        self.signature = digest_obj(sig_spec)
        write_json(self.out / "run_specification.json", self.spec)
        write_json(self.out / "preflight.json", {"errors": self.preflight_errors,
            "signature": self.signature, "status": "failed" if self.preflight_errors else "complete"})
        if self.preflight_errors:
            raise RuntimeError("Preflight found all listed blockers; see preflight.json. No GPU inference started.")

    def load_maps(self, domain, cfg):
        native_path = self.repo / "configs" / f"mapping_{domain}_to_common.yaml"
        train_path = self.repo / "configs" / ("train_id_to_common_id.yaml" if domain == "eclair"
                                             else "train_id_to_common_id_dales.yaml")
        # Resolve actual mapping filename from the canonical evaluation, never infer class order.
        example = next(e for e in self.evidence["evaluations"] if e["source"] == domain)
        train_path = self.repo / example["train_mapping_path"]
        native_path = self.repo / example["native_mapping_path"]
        nat = self.ev.mapping_to_lut(self.ev.load_yaml_int_map(native_path), size=256, default=0, name="native")
        tr = self.ev.mapping_to_lut(self.ev.load_yaml_int_map(train_path),
            size=int(cfg["model"]["out_channels"]), default=0, name="train")
        self.ev.validate_source_train_to_common_mapping(source_cfg=cfg, source_domain=domain,
            source_train_to_common=tr, source_native_to_common_lut=nat, logger=self.log)
        nt, _ = self.ev.source_native_to_train_lut_from_config(source_cfg=cfg,
            source_domain=domain, logger=self.log)
        self.spec.setdefault("mapping_hashes", {})[str(native_path)] = digest_file(native_path)
        self.spec["mapping_hashes"][str(train_path)] = digest_file(train_path)
        self.spec.setdefault("label_maps", {})[domain] = {"native_to_common": nat,
            "train_to_common": tr, "native_to_train": nt,
            "source_train_names": self.ev.get_train_class_names(cfg, int(cfg["model"]["out_channels"]))}
        return {"native_common": nat, "train_common": tr, "native_train": nt}

    def tile_identities(self, ds, domain):
        rows = []
        mod = importlib.import_module("src.data_" + domain)
        members = ds.names if domain == "eclair" else ds.files
        for i, member in enumerate(members):
            key_error = None
            if domain == "eclair":
                try:
                    path = mod._resolve_pc_path(ds.eclair_root, member)
                except FileNotFoundError:
                    path = ds.eclair_root / "pointclouds" / member
                try:
                    cache = ds._cache_path(member)
                except OSError as exc:
                    cache, key_error = None, str(exc)
            else:
                path = member
                try:
                    cache = ds._cache_dir / (mod._cache_key_for_dales_raw(path=path) + ".pt") if ds.use_cache else None
                except OSError as exc:
                    cache, key_error = None, str(exc)
            rows.append({"index": i, "raw_file": file_identity(path),
                         "cache": file_identity(cache) if cache else None, "cache_key_error": key_error})
        return rows

    def labels(self, raw, domain):
        native = self.ev.get_native_labels(raw, target_domain=domain)
        if native.ndim != 1 or np.any((native < 0) | (native >= 256)):
            raise RuntimeError("Native label out of LUT range")
        return native, self.luts[domain]["native_common"][native], self.luts[domain]["native_train"][native]

    def scan(self, domain, split, index):
        raw = self.ds[domain][split].get_raw(index)
        xyz = np.asarray(raw["xyz"], np.float64)
        native, common, train = self.labels(raw, domain)
        if xyz.shape != (len(native), 3) or not np.isfinite(xyz).all():
            raise RuntimeError("Nonfinite coordinates or point/label cardinality mismatch")
        if len(xyz) == 0:
            raise RuntimeError("Empty tile")
        span = np.ptp(xyz, axis=0)
        result = {"domain": domain, "split": split, "index": index,
            "native_histogram": np.bincount(native, minlength=256),
            "common_histogram": np.bincount(common, minlength=8), "points": len(xyz),
            "span_m": span, "bbox_points_per_xy_m2": len(xyz)/max(1e-12, span[0]*span[1]),
            "numeric_payload_sha256": payload_digest(raw), "returns": {}, "voxel_diagnostics": []}
        for key in ("return_number", "number_of_returns"):
            value = raw.get(key)
            if value is None:
                result["returns"][key] = {"missing": True}
                continue
            value = np.asarray(value)
            if value.shape != native.shape or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid {key} shape or values")
            encoded = np.clip(value.astype(np.int64), 1, 5) - 1
            result["returns"][key] = {"min": float(value.min()), "max": float(value.max()),
                "clipped_points": int(((value < 1) | (value > 5)).sum()),
                "common_class_by_encoded_bin": np.bincount(common*5+encoded, minlength=40).reshape(8, 5)}
        if raw.get("return_number") is not None and raw.get("number_of_returns") is not None:
            rn, nr = np.asarray(raw["return_number"]), np.asarray(raw["number_of_returns"])
            joint = common*25 + (np.clip(rn.astype(np.int64), 1, 5)-1)*5 + np.clip(nr.astype(np.int64), 1, 5)-1
            result["returns"]["joint_common_class_by_rn_nor"] = np.bincount(joint, minlength=200).reshape(8, 5, 5)
            result["returns"]["rn_exceeds_nor_points"] = int((rn > nr).sum())
        for edge in self.args.voxel_edges:
            result["voxel_diagnostics"].append(voxel_diagnostic(xyz, common, train,
                self.luts[domain]["train_common"], edge))
        return result

    def make_patch(self, domain, split, cls):
        ds = self.ds[domain][split]
        order = np.random.default_rng(stable_seed(self.args.seed, domain, split)).permutation(len(ds))
        selected = None
        for i in order:
            stage = self.out / "stages" / f"scan_{domain}_{split}_{i:04d}.json"
            if not stage.is_file():
                continue
            record = json.loads(stage.read_text())
            if record.get("signature") == self.signature and record.get("status") == "complete" and record["result"]["common_histogram"][cls]:
                selected = int(i)
                break
        if selected is None:
            return {"available": False, "reason": "No focal-class points in successful split scan", "class": cls}
        raw = ds.get_raw(selected)
        scan = json.loads((self.out / "stages" / f"scan_{domain}_{split}_{selected:04d}.json").read_text())
        if payload_digest(raw) != scan["result"]["numeric_payload_sha256"]:
            raise RuntimeError("Dataset payload changed after its scan")
        xyz = raw["xyz"]
        _, common, _ = self.labels(raw, domain)
        rng = np.random.default_rng(stable_seed(self.args.seed, domain, split, selected, cls))
        center = xyz[int(rng.choice(np.flatnonzero(common == cls)))]
        keep = np.all(np.abs(xyz[:, :2] - center[None, :2]) <= self.args.patch_m/2, axis=1)
        subset = subset_raw(raw, keep, relocalize=True)
        path = self.out / "working_patches" / self.signature[:16] / f"{domain}_{split}_{cls}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in subset.items()
            if k in ("xyz", "native_labels", "return_number", "number_of_returns", "intensity", "rgb")
            and v is not None})
        return {"available": True, "artifact_path": str(path), "artifact_sha256": digest_file(path),
            "origin_tile_index": selected, "center_local_m": center, "class": cls,
            "patch_m": self.args.patch_m, "points": int(keep.sum())}

    def infer(self, model, raw, run, domain, *, window=120., cap=180000, capture=False, phase_shift=False):
        """Observe actual calls without replacing scheduling, quantization or prediction."""
        ev, torch = self.ev, self.torch
        original_forward, original_voxel = ev.run_model_on_voxel_subset, ev.voxelize_from_q
        state = {"leaves": [], "seen_feature_voxels": set(), "features": [], "oom_attempts": 0,
                 "normalization_observations": {}}
        native, common, _ = self.labels(raw, domain)
        meters = run["config"]["data"]["patch"]["voxel_size"] * run["config"]["data"]["patch"]["coord_norm_factor"]
        state["meters"] = meters

        def phase_shifted_model(st):
            # Identical active set/features and physical relative geometry; only the
            # coordinate phase seen by strided sparse layers changes. Map back to
            # the input coordinate map so the canonical alignment remains valid.
            shifted_coords = st.C.to(self.device).int() + torch.tensor([0, 1, 1, 0],
                dtype=torch.int32, device=self.device)
            shifted = ev.ME.SparseTensor(features=st.F, coordinates=shifted_coords, device=self.device)
            shifted_out = ev.sparse_output_from_model_output(model(shifted))
            aligned = ev.align_sparse_output_to_input_coords(out=shifted_out,
                coords_in_t=shifted_coords, device=self.device)
            return ev.ME.SparseTensor(features=aligned, coordinate_map_key=st.coordinate_map_key,
                coordinate_manager=st.coordinate_manager)

        def observe_voxel(**kw):
            vx = original_voxel(**kw)
            state["inverse"] = vx["inverse_map"]
            state["coords"] = vx["coords_u"]
            inv = state["inverse"]
            if capture:
                maj, _, _, _ = grouped_majority(inv, common, 8, ignore=0)
                chosen = []
                rng = np.random.default_rng(stable_seed(self.args.seed, run["run"], domain, len(common)))
                for cls in range(1, 8):
                    ids = np.flatnonzero(maj == cls)
                    if len(ids):
                        chosen.extend(rng.choice(ids, min(64, len(ids)), replace=False).tolist())
                state["feature_probe_ids"] = np.asarray(sorted(chosen), np.int64)
                state["majority"] = maj
            return vx

        def observe_forward(**kw):
            sel, coords = kw["sel"], kw["coords_u"]
            sub = coords[sel]
            minimum, maximum = sub.min(0), sub.max(0)
            state["current"] = (sel, minimum)
            state["pending_features"] = []
            state["pending_normalization"] = []
            tick = time.perf_counter()
            try:
                if phase_shift:
                    kw = dict(kw, model=phase_shifted_model)
                result = original_forward(**kw)
            except Exception as exc:
                state["pending_features"] = []
                state["pending_normalization"] = []
                if ev.is_oom_exception(exc):
                    state["oom_attempts"] += 1
                raise
            for voxel, cls, value in state["pending_features"]:
                state["seen_feature_voxels"].add(voxel)
                state["features"].append((cls, value))
            for name, n, total, total_sq in state["pending_normalization"]:
                current = state["normalization_observations"].setdefault(name,
                    {"n": 0, "sum": np.zeros_like(total), "sum_sq": np.zeros_like(total_sq)})
                current["n"] += n
                current["sum"] += total
                current["sum_sq"] += total_sq
            state["leaves"].append({"n_voxels": len(sel), "min_q": minimum,
                "max_q": maximum, "span_m": (maximum-minimum+1)*meters,
                "downsampling_phase_mod16": minimum % 16,
                "seconds": time.perf_counter()-tick})
            # Bounding-box proximity is a proxy, not exact physical split-plane distance.
            if "boundary" not in state:
                state["boundary"] = np.full(len(coords), np.inf, np.float32)
            dist = np.minimum(sub[:, :2]-minimum[:2], maximum[:2]-sub[:, :2]).min(1) * meters
            state["boundary"][sel] = np.minimum(state["boundary"][sel], dist)
            return result

        def feature_hook(_module, _inputs, output):
            if not capture or "current" not in state:
                return
            sel, minimum = state["current"]
            probes = state.get("feature_probe_ids", np.empty(0, np.int64))
            positions = np.flatnonzero(np.isin(sel, probes))
            positions = np.asarray([i for i in positions if int(sel[i]) not in state["seen_feature_voxels"]], np.int64)
            if not len(positions):
                return
            coords = state["coords"][sel] - minimum
            batched = np.column_stack((np.zeros(len(coords), np.int32), coords)).astype(np.int32)
            aligned = ev.align_sparse_output_to_input_coords(out=output,
                coords_in_t=torch.from_numpy(batched).int().to(self.device), device=self.device)
            selected = aligned[torch.as_tensor(positions, device=self.device)].detach().float().cpu().numpy()
            for i, value in zip(positions, selected):
                voxel = int(sel[i])
                state["pending_features"].append((voxel, int(state["majority"][voxel]), value))

        handles = []
        if capture:
            modules = [m for n, m in model.named_modules() if n.split(".")[-1] == "block8"]
            if len(modules) != 1:
                raise RuntimeError("Expected exactly one block8 feature tap; no guessed fallback")
            handles.append(modules[0].register_forward_hook(feature_hook))
            bns = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            selected_bns = bns[:2] + [(n, m) for n, m in bns if "block8" in n]
            for name, bn in dict(selected_bns).items():
                def pre_bn(module, inputs, name=name):
                    value = inputs[0].detach()
                    step = max(1, len(value)//512)
                    sample = value[::step][:512].float()
                    if len(sample):
                        state["pending_normalization"].append((name, len(sample),
                            sample.sum(0).cpu().numpy().astype(np.float64),
                            (sample*sample).sum(0).cpu().numpy().astype(np.float64)))
                handles.append(bn.register_forward_pre_hook(pre_bn))
        ev.run_model_on_voxel_subset, ev.voxelize_from_q = observe_forward, observe_voxel
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            logits = ev.infer_tile_point_logits(model=model, raw=raw, source_cfg=run["config"],
                device=self.device, amp=False, window_size_m=window, window_stride_m=window,
                aggregation="mean_logits", max_voxels_per_forward=cap, max_split_depth=8,
                logger=self.log, target_domain=domain)
            torch.cuda.synchronize()
        finally:
            ev.run_model_on_voxel_subset, ev.voxelize_from_q = original_forward, original_voxel
            for handle in handles:
                handle.remove()
        state["seconds"] = time.perf_counter()-start
        state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        state["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return logits, state

    def evaluate_tile(self, model, run, domain, index):
        raw = self.ds[domain]["test"].get_raw(index)
        scan_path = self.out / "stages" / f"scan_{domain}_test_{index:04d}.json"
        scan = json.loads(scan_path.read_text())
        if scan.get("status") != "complete" or scan.get("signature") != self.signature:
            raise RuntimeError("Canonical inference requires a successful matching tile scan")
        if payload_digest(raw) != scan["result"]["numeric_payload_sha256"]:
            raise RuntimeError("Test data changed during the collection")
        native, gt, _ = self.labels(raw, domain)
        logits, state = self.infer(model, raw, run, domain, capture=True)
        pred = logits.argmax(1)
        tr = self.luts[run["source"]]["train_common"]
        common_pred = tr[pred]
        merged = np.zeros((len(logits), 8), np.float32)
        # Stable probability-sum diagnostic; does not replace native argmax.
        probability = np.exp(logits - logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        for i, cls in enumerate(tr):
            merged[:, cls] += probability[:, i]
        restricted = logits.copy()
        restricted[:, tr == 0] = -np.inf
        merged_valid_only = merged.copy()
        merged_valid_only[:, 0] = -np.inf
        inv, coords = state["inverse"], state["coords"]
        point_boundary = state["boundary"][inv]
        supported = gt != 0
        confidence = {}
        for cls in range(1, 8):
            mask = gt == cls
            if not mask.any():
                continue
            margin = logits[mask][:, tr == cls].max(1) - logits[mask][:, tr != cls].max(1)
            confidence[cls] = {"best_correct_common_native_logit_minus_best_other_quantiles":
                np.quantile(margin, [0., .1, .5, .9, 1.]),
                "maximum_native_probability_quantiles": np.quantile(probability[mask].max(1), [0., .1, .5, .9, 1.])}
        feature_stats = {}
        for cls in range(1, 8):
            a = np.asarray([v for c, v in state["features"] if c == cls], np.float64)
            feature_stats[cls] = {"n": len(a), "sum": a.sum(0), "sum_sq": (a*a).sum(0)} if len(a) else {"n": 0}
        occupancy = np.bincount(inv, minlength=len(coords))
        strata = {}
        for label, mask in (("leaf_bbox_within_2m", point_boundary < 2),
                            ("leaf_bbox_interior_ge2m", point_boundary >= 2),
                            ("singleton_voxel", occupancy[inv] == 1),
                            ("multi_point_voxel", occupancy[inv] > 1)):
            strata[label] = metrics(cm_from(gt[mask], common_pred[mask]))
        examples = []
        rng = np.random.default_rng(stable_seed(self.args.seed, run["run"], domain, index))
        for cls in range(1, 8):
            for correct in (False, True):
                ids = np.flatnonzero((gt == cls) & ((common_pred == gt) == correct))
                chosen = rng.choice(ids, min(8, len(ids)), replace=False)
                for j in chosen:
                    examples.append({"raw_point_index": int(j), "xyz_local_m": raw["xyz"][j],
                        "gt_native": int(native[j]), "gt_common": cls, "pred_train": int(pred[j]),
                        "pred_common": int(common_pred[j]), "native_logits": logits[j]})
        return {"run": run["run"], "source": run["source"], "target": domain,
            "tile_index": index, "n_points_total": len(gt), "metrics": metrics(cm_from(gt, common_pred)),
            "common_gt_by_native_prediction": cm_from(gt, pred, columns=len(tr)),
            "target_native_gt_by_source_train_prediction": np.bincount(native*len(tr)+pred,
                minlength=256*len(tr)).reshape(256, len(tr)),
            "class_confidence_diagnostics": confidence,
            "probability_sum_decision_diagnostic": metrics(cm_from(gt, merged.argmax(1))),
            "probability_sum_exclude_non_common_diagnostic": metrics(cm_from(gt, merged_valid_only.argmax(1))),
            "exclude_non_common_output_diagnostic": metrics(cm_from(gt, tr[restricted.argmax(1)])),
            "mean_non_common_probability_by_gt": [float(merged[gt == c, 0].mean()) if np.any(gt == c) else None for c in range(8)],
            "strata": strata, "feature_moments": feature_stats, "examples": examples,
            "normalization_observations": state["normalization_observations"],
            "leaf_calls": state["leaves"], "oom_attempts": state["oom_attempts"],
            "runtime_s": state["seconds"], "peak_cuda_allocated_bytes": state["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": state["peak_cuda_reserved_bytes"],
            "valid_points": int(supported.sum()),
            "timing_scope": "Instrumented inference, including feature observation; not a production latency benchmark.",
            "feature_sampling": "Up to 64 common-majority voxels/class/tile; first successful leaf context. Descriptive, not a causal separability test."}

    def probes(self, model, run, domain, split, cls):
        p = self.out / "working_patches" / self.signature[:16] / f"{domain}_{split}_{cls}.npz"
        if not p.is_file():
            return {"available": False, "run": run["run"], "domain": domain, "split": split,
                "focal_class": cls, "reason": "No class patch: absent coverage is not a negative finding"}
        with np.load(p) as obj:
            raw = {k: obj[k] for k in obj.files if not k.startswith("_")}
        _, gt, _ = self.labels(raw, domain)
        tr = self.luts[run["source"]]["train_common"]
        logits, state = self.infer(model, raw, run, domain, window=self.args.patch_m)
        prediction = tr[logits.argmax(1)]
        report = {"run": run["run"], "domain": domain, "split": split, "focal_class": cls,
            "available": True,
            "point_count": len(gt), "clean": metrics(cm_from(gt, prediction)), "variants": {},
            "interpretation": "Frozen-weight sensitivity only. No intervention identifies a training-time cause."}
        rng = np.random.default_rng(stable_seed(self.args.seed, domain, split, cls))
        keep = rng.random(len(gt)) < .5
        keep[np.argmin(raw["xyz"], axis=0)] = True  # Preserve clean quantization origin.
        shuffled = copy.copy(raw)
        perm = rng.permutation(len(gt))
        for key in ("return_number", "number_of_returns"):
            shuffled[key] = raw[key][perm]
        constant = copy.copy(raw)
        for key in ("return_number", "number_of_returns"):
            constant[key] = np.ones_like(raw[key])
        q = quantize_like_evaluator(raw["xyz"], state["meters"])
        _, inverse = np.unique(q, axis=0, return_inverse=True)
        retain_voxels = rng.random(int(inverse.max())+1) >= .1
        retain_voxels[inverse[np.argmin(raw["xyz"], axis=0)]] = True
        keep_vox = retain_voxels[inverse]
        cases = [("return_pair_shuffle", shuffled, np.ones(len(gt), bool), self.args.patch_m),
                 ("constant_returns_inference_ood", constant, np.ones(len(gt), bool), self.args.patch_m),
                 ("raw_point_thin_50pct", subset_raw(raw, keep), keep, self.args.patch_m),
                 ("active_voxel_thin_10pct_unprotected", subset_raw(raw, keep_vox), keep_vox, self.args.patch_m),
                 ("half_window_context_and_phase_joint", raw, np.ones(len(gt), bool), self.args.patch_m/2),
                 ("fixed_context_stride_phase_shift_xy_one_voxel", raw, np.ones(len(gt), bool), self.args.patch_m)]
        for name, variant, mask, win in cases:
            out, st = self.infer(model, variant, run, domain, window=win,
                phase_shift=(name == "fixed_context_stride_phase_shift_xy_one_voxel"))
            changed_pred = tr[out.argmax(1)]
            report["variants"][name] = {"metrics": metrics(cm_from(gt[mask], changed_pred)),
                "paired_clean_same_retained_points": metrics(cm_from(gt[mask], prediction[mask])),
                "prediction_disagreement_fraction": float((prediction[mask] != changed_pred).mean()),
                "retained_class_points": np.bincount(gt[mask], minlength=8),
                "leaf_calls": st["leaves"], "seconds": st["seconds"]}
        report["geometry"] = patch_geometry(raw["xyz"], gt, seed=stable_seed(domain, split, cls))
        _, common_labels, native_train = self.labels(raw, domain)
        report["voxel_phase_diagnostics"] = []
        for edge in (.2, .5):
            for phase in (0., .5):
                vd = voxel_diagnostic(raw["xyz"] + phase*edge, common_labels, native_train,
                    self.luts[domain]["train_common"], edge)
                vd["grid_offset_fraction_xyz"] = phase
                report["voxel_phase_diagnostics"].append(vd)
        return report

    def replay_training(self, run):
        module_spec = importlib.util.spec_from_file_location("diagnostic_project_train", self.repo / "train.py")
        train_module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = train_module
        module_spec.loader.exec_module(train_module)
        cfg = copy.deepcopy(run["config"])
        cfg["data"]["num_workers"] = 0
        cfg["data"]["write_cache"] = False
        env = importlib.import_module("src.dist").DistEnv(enabled=False)
        loader, _, _ = train_module.build_dataloaders(cfg, env)
        ds = loader.dataset
        samples = []
        # Epochs are embedded from the actual configuration at packaging time.
        last = int(run["epochs"])
        for epoch in sorted(set((1, max(1, last//2), last))):
            ds.set_epoch(epoch)
            rng = np.random.default_rng(stable_seed(self.args.seed, run["run"], epoch))
            indices = rng.choice(len(ds), min(len(ds), self.args.training_items), replace=False)
            for index in indices:
                voxel_calls = []
                data_modules = [importlib.import_module("src.data_eclair"), importlib.import_module("src.data_dales")]
                originals = [m.voxelize_from_q for m in data_modules]
                def observe_training_voxel(**kw):
                    vx = originals[0](**kw)
                    yp, yu = kw.get("labels_p_i64"), vx.get("labels_u")
                    voxel_calls.append({"input_points": len(kw["q_int32"]),
                        "raw_train_label_histogram": np.bincount(yp[yp >= 0], minlength=int(cfg["model"]["out_channels"])) if yp is not None else None,
                        "output_voxels_before_later_caps": len(vx["coords_u"]),
                        "voxel_train_label_histogram_before_later_caps": np.bincount(yu[yu >= 0], minlength=int(cfg["model"]["out_channels"])) if yu is not None else None})
                try:
                    for mod in data_modules:
                        mod.voxelize_from_q = observe_training_voxel
                    try:
                        item = ds[int(index)]
                    finally:
                        for mod, original in zip(data_modules, originals):
                            mod.voxelize_from_q = original
                    for sample in (item if isinstance(item, list) else [item]):
                        coords, labels = sample["coords"], sample["labels"]
                        y = labels.numpy()
                        mapped = np.zeros(len(y), np.int64)
                        valid = y >= 0
                        mapped[valid] = self.luts[run["source"]]["train_common"][y[valid]]
                        r = {"epoch": epoch, "index": int(index), "voxels": len(y),
                            "common_voxel_labels": np.bincount(mapped, minlength=8),
                            "native_train_voxel_labels": np.bincount(y[valid], minlength=int(cfg["model"]["out_channels"])),
                            "span_m": np.ptp(coords.numpy()[:, -3:], axis=0) *
                                cfg["data"]["patch"]["voxel_size"] * cfg["data"]["patch"]["coord_norm_factor"],
                            "metadata": {k: v.detach().cpu().numpy() for k, v in sample.items()
                                         if k.startswith("meta_") and self.torch.is_tensor(v)},
                            "voxelization_candidate_calls": voxel_calls,
                            "candidate_call_scope": "Includes rejected/resampled candidates; do not sum as accepted exposure."}
                        oc = importlib.import_module("src.ocons")
                        ocfg = oc.OConsConfig.from_cfg(cfg)
                        if ocfg.enabled:
                            gen = self.torch.Generator().manual_seed(stable_seed(self.args.seed, epoch, int(index)))
                            batched = self.torch.cat((self.torch.zeros((len(coords), 1), dtype=coords.dtype), coords), dim=1)
                            view = oc.make_masked_view(batched, sample["feats"], labels, ocfg,
                                num_classes=int(cfg["model"]["out_channels"]), generator=gen)
                            kept = view.labels.numpy()
                            r["ocons_retained_native_histogram"] = np.bincount(kept[kept >= 0], minlength=int(cfg["model"]["out_channels"]))
                            r["ocons_actual_mask_fraction"] = view.diagnostics.actual_mask_fraction
                        samples.append(clean_json(r))
                except Exception as exc:
                    samples.append({"epoch": epoch, "index": int(index), "error": str(exc)})
        errors = [s for s in samples if "error" in s]
        result = {"run": run["run"], "samples": samples, "item_errors": errors,
            "scope": "Seeded single-process replay of current loaders at 3 epochs; not reconstruction of historical DDP exposure or optimization."}
        if errors:
            write_json(self.out / "training_replay_errors" / (run["run"] + ".json"), result)
            raise RuntimeError(f"{len(errors)} training replay items failed; all item errors preserved")
        return result

    def execute(self):
        self.signature = digest_obj(self.spec)
        self.task("offline_evidence_audit", lambda: offline_audit(self.evidence))
        if self.args.audit_only:
            self.workflow_completed = True
            return
        self.prepare()
        if self.args.preflight_only:
            self.workflow_completed = True
            return
        for domain in ("eclair", "dales"):
            for split in ("train", "val", "test"):
                ds = self.ds[domain][split]
                order = np.random.default_rng(stable_seed(self.args.seed, domain, split)).permutation(len(ds))
                for index in order:
                    self.task(f"scan_{domain}_{split}_{index:04d}",
                              lambda d=domain, s=split, i=int(index): self.scan(d, s, i))
                if split in ("val", "test"):
                    for cls in range(1, 8):
                        self.task(f"cohort_{domain}_{split}_{cls}",
                            lambda d=domain, s=split, c=cls: self.make_patch(d, s, c))
        for run in self.runs:
            self.task(f"training_replay_{run['run']}", lambda r=run: self.replay_training(r))
            def checkpoint_work(r=run):
                if digest_file(r["checkpoint"]) != r["checkpoint_identity"]["sha256"]:
                    raise RuntimeError("Checkpoint changed during data collection")
                checkpoint = self.torch.load(r["checkpoint"], map_location="cpu")
                checkpoint_metadata = {}
                if isinstance(checkpoint, dict):
                    for key, value in checkpoint.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            checkpoint_metadata[key] = value
                        elif self.torch.is_tensor(value) and value.numel() == 1:
                            checkpoint_metadata[key] = value.item()
                        elif key in ("cfg", "config", "configuration") and isinstance(value, dict):
                            checkpoint_metadata[key] = value
                write_json(self.out / "checkpoint_metadata" / (r["run"] + ".json"), checkpoint_metadata)
                del checkpoint
                model = self.ev.load_model_for_eval(source_cfg=r["config"], ckpt_path=r["checkpoint"],
                                                    device=self.device, logger=self.log)
                before = bn_digest(model)
                architecture = {"run": r["run"], "module": type(model).__name__,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "normalization": [{"name": n, "type": type(m).__name__,
                        "momentum": m.momentum, "eps": m.eps,
                        "running_mean": m.running_mean.detach().cpu().numpy() if m.running_mean is not None else None,
                        "running_var": m.running_var.detach().cpu().numpy() if m.running_var is not None else None}
                        for n, m in model.named_modules() if isinstance(m, self.torch.nn.modules.batchnorm._BatchNorm)]}
                for domain in ("eclair", "dales"):
                    for index in range(len(self.ds[domain]["test"])):
                        self.task(f"eval_{r['run']}_{domain}_{index:04d}",
                            lambda d=domain, i=index: self.evaluate_tile(model, r, d, i))
                    for split in (("val", "test") if domain == r["source"] else ("test",)):
                        for cls in range(1, 8):
                            self.task(f"probe_{r['run']}_{domain}_{split}_{cls}",
                                lambda d=domain, s=split, c=cls: self.probes(model, r, d, s, c))
                if bn_digest(model) != before:
                    raise RuntimeError("Frozen inference modified normalization buffers")
                del model
                self.torch.cuda.empty_cache()
                return architecture
            self.task("checkpoint_" + run["run"], checkpoint_work, resume=False)
        self.workflow_completed = self.task("aggregate", self.aggregate, resume=False) is not None

    def aggregate(self):
        records = {}
        scans, probes, training_replays = [], [], []
        for p in (self.out / "stages").glob("*.json"):
            x = json.loads(p.read_text())
            if x.get("status") != "complete" or x.get("signature") != self.signature:
                continue
            if p.name.startswith("scan_"):
                scans.append(x["result"])
            elif p.name.startswith("probe_"):
                probes.append(x["result"])
            elif p.name.startswith("training_replay_"):
                training_replays.append(x["result"])
        for p in (self.out / "stages").glob("eval_*.json"):
            x = json.loads(p.read_text())
            if x.get("status") != "complete" or x.get("signature") != self.signature:
                continue
            r = x["result"]
            records.setdefault((r["run"], r["target"]), []).append(r)
        totals, comparisons = [], []
        for entry in self.evidence["evaluations"]:
            rows = sorted(records.get((entry["run"], entry["target"]), []), key=lambda x: x["tile_index"])
            cm = sum((np.asarray(x["metrics"]["cm"], np.int64) for x in rows), np.zeros((8, 8), np.int64))
            expected = np.asarray(entry["cm"], np.int64)
            totals.append({"run": entry["run"], "source": entry["source"], "target": entry["target"],
                "completed_tiles": len(rows), "expected_tiles": entry["n_tiles"],
                "complete_coverage": len(rows) == entry["n_tiles"], "metrics": metrics(cm),
                "exact_historical_confusion_match": bool(np.array_equal(cm, expected)),
                "changed_confusion_count_l1": int(np.abs(cm-expected).sum()),
                "native_prediction_destinations": sum((np.asarray(x["common_gt_by_native_prediction"], np.int64)
                    for x in rows), np.zeros((8, 11 if entry["source"] == "eclair" else 8), np.int64))})
        for source in ("eclair", "dales"):
            base = next(r for r in self.runs if r["source"] == source and r["method"] == "M0")
            for run in self.runs:
                if run["source"] != source or run["method"] == "M0":
                    continue
                for domain in ("eclair", "dales"):
                    a = {x["tile_index"]: x for x in records.get((base["run"], domain), [])}
                    b = {x["tile_index"]: x for x in records.get((run["run"], domain), [])}
                    shared = sorted(a.keys() & b.keys())
                    if len(shared) != EXPECTED_TILES[domain]["test"]:
                        comparisons.append({"run": run["run"], "target": domain, "status": "incomplete"})
                        continue
                    aa = np.asarray([a[i]["metrics"]["cm"] for i in shared], np.int64)
                    bb = np.asarray([b[i]["metrics"]["cm"] for i in shared], np.int64)
                    if not np.array_equal(aa.sum(2), bb.sum(2)):
                        raise RuntimeError("Paired tile ground-truth supports differ across checkpoints")
                    comparisons.append({"run": run["run"], "target": domain,
                        "paired_tile_bootstrap": bootstrap_delta(aa, bb, stable_seed(run["run"], domain)),
                        "warning": "Conditional on these checkpoints and tiles; not seed or geographic-domain uncertainty."})
        summaries = summarize_diagnostics(scans, probes, training_replays, records)
        historical = []
        for r in self.runs:
            if not r.get("historical_comparator"):
                continue
            for domain in ("eclair", "dales"):
                rows = records.get((r["run"], domain), [])
                cm = sum((np.asarray(x["metrics"]["cm"], np.int64) for x in rows), np.zeros((8, 8), np.int64))
                historical.append({"run": r["run"], "method": r["method"], "source": r["source"],
                    "target": domain, "metrics": metrics(cm), "completed_tiles": len(rows),
                    "expected_tiles": EXPECTED_TILES[domain]["test"],
                    "scope": "Exploratory existing-checkpoint comparator; historical execution source is not certified. Inspect configuration differences before attribution."})
        report = {"evaluations": totals, "historical_comparators": historical,
            "paired_method_comparisons": comparisons,
            **summaries,
            "decision_rules": ["Historical confusion mismatch => resolve identity/protocol before attributing gains.",
                "Voxel impurity => quantify affected-class limitations; no mIoU-ceiling or root-cause claim.",
                "Decision-rule changes => evidence of sensitivity, not proof a seven-class training head will help.",
                "Frozen perturbation sensitivity => testable hypothesis, not proof that feature removal or regularization will train better.",
                "Architecture capacity remains unidentified until matched trained alternatives exist."],
            "unavailable_by_design": ["Historical exact per-step exposure", "Independent seed variance",
                "Performance of untrained architectures", "Causal ranking of all future methods", "Guaranteed solution"]}
        write_json(self.out / "diagnostic_results.json", report)
        lines = ["# Diagnostic measurements and decision boundaries", "",
            "These measurements prioritize controlled experiments; they do not establish a unique cause.", "",
            "## Reproduction", "", "| Run | Source | Target | Tiles | Exact historical CM | mIoU |",
            "|---|---|---|---:|---|---:|"]
        for e in totals:
            lines.append(f"| {e['run']} | {e['source']} | {e['target']} | {e['completed_tiles']}/{e['expected_tiles']} | {e['exact_historical_confusion_match']} | {e['metrics']['legacy_miou_all_seven']:.6f} |")
        lines += ["", "## Read the measurements in this order", "",
            "1. Resolve any reproduction or coverage discrepancy before attributing gains.",
            "2. dataset_geometry: compare weak-class label survival and oracle back-projections, not only the globally dominated accuracy ceiling.",
            "3. Native prediction destinations and decision-rule scores: separate the rejection sink from ordinary semantic confusions.",
            "4. training_exposure: inspect selected voxel labels, actual crop sizes and class absence. Candidate attempts are not accepted training exposure.",
            "5. perturbation_summaries: compare each perturbation with clean predictions on the identical retained points.",
            "6. feature_domain_comparison: use class-conditioned descriptive shifts as clues, not proof that alignment will help.",
            "7. Use the predeclared decision table in the accompanying review to choose one bounded training comparison.", "",
            "Context probes jointly alter support and sparse-grid phase. Return ablations are inference-time OOD tests.",
            "Class-centred patches may overlap. Their aggregate is not a test-set score or an independent-sample significance estimate."]
        (self.out / "MEASUREMENTS_AND_DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")
        return report

    def finish(self):
        status = "interrupted" if self.interrupted else "failed" if self.failures else ("audit_only" if self.args.audit_only else
                 "preflight_only" if self.args.preflight_only else "complete")
        if not self.workflow_completed and status == "complete":
            status = "incomplete"
        p = self.out / "diagnostic_results.json"
        if p.is_file() and status == "complete":
            evaluations = json.loads(p.read_text())["evaluations"]
            if not all(x["complete_coverage"] for x in evaluations):
                status = "incomplete"
            elif not all(x["exact_historical_confusion_match"] for x in evaluations):
                status = "complete_with_reproduction_discrepancies"
        write_json(self.out / "STATUS.json", {"status": status, "events": self.events,
            "failures": self.failures, "scientific_scope": "Bounded diagnostics, not training or proof of a solution"})
        summary = ["# One-shot diagnosis", "", f"Status: **{status}**", "",
            "Read STATUS.json first. diagnostic_results.json contains complete-coverage and exact-reproduction checks.",
            "Stage records preserve class-specific counts, native rejection destinations, voxel limits,",
            "paired frozen-weight perturbations, real forward extents/phases, sampled features, and loader replay.",
            "Working patches are excluded from the upload bundle. No checkpoint weights or full dataset are bundled.",
            "", "This run does not train unseen architectures or reconstruct missing historical measurements."]
        (self.out / "READ_RESULTS.md").write_text("\n".join(summary), encoding="utf-8")
        files = [p for p in self.out.rglob("*") if p.is_file() and "working_patches" not in p.parts
                 and p.name != "bundle_manifest.json"
                 and p.suffix != ".gz" and not p.name.endswith(".tmp")]
        write_json(self.out / "bundle_manifest.json", [{"path": p.relative_to(self.out).as_posix(),
            "sha256": digest_file(p), "bytes": p.stat().st_size} for p in files if p.name != "bundle_manifest.json"])
        archive = self.out.parent / (self.out.name + "_results.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for p in files + [self.out / "bundle_manifest.json"]:
                if p.is_file():
                    tar.add(p, arcname=p.relative_to(self.out).as_posix())
        self.log.info("RESULT %s ; status=%s", archive, status)


def subset_raw(raw, mask, relocalize=False):
    n = len(raw["xyz"])
    out = {k: (np.asarray(v)[mask] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
           for k, v in raw.items()}
    if relocalize:
        out["xyz"] = out["xyz"].astype(np.float64) - out["xyz"].min(0)
    return out


def bn_digest(model):
    h = hashlib.sha256()
    for name, value in model.named_buffers():
        if any(key in name for key in ("running_mean", "running_var", "num_batches_tracked")):
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def summarize_diagnostics(scans, probes, replays, records):
    geometry = []
    for domain in ("eclair", "dales"):
        for split in ("train", "val", "test"):
            rows = [r for r in scans if r["domain"] == domain and r["split"] == split]
            hist = sum((np.asarray(r["common_histogram"], np.int64) for r in rows), np.zeros(8, np.int64))
            edges = sorted({v["edge_m"] for r in rows for v in r["voxel_diagnostics"]})
            for edge in edges:
                vox = [v for r in rows for v in r["voxel_diagnostics"] if v["edge_m"] == edge]
                contains = sum((np.asarray(v["class_containing_voxels"], np.int64) for v in vox), np.zeros(8, np.int64))
                survives = sum((np.asarray(v["class_native_majority_then_map_voxels"], np.int64) for v in vox), np.zeros(8, np.int64))
                oracle = sum((np.asarray(v["common_majority_backprojection"]["cm"], np.int64) for v in vox), np.zeros((8, 8), np.int64))
                total = sum(v["valid_common_points"] for v in vox)
                geometry.append({"domain": domain, "split": split, "edge_m": edge,
                    "tiles": len(rows), "expected_tiles": EXPECTED_TILES[domain][split],
                    "raw_common_histogram": hist, "class_containing_voxels": contains,
                    "native_majority_label_survival_fraction": np.divide(survives, contains,
                        out=np.zeros(8), where=contains > 0),
                    "common_majority_backprojection": metrics(oracle),
                    "point_weighted_accuracy_ceiling": sum(v["constant_voxel_accuracy_ceiling"] * v["valid_common_points"] for v in vox)/max(1, total),
                    "return_histograms": {key: sum((np.asarray(r["returns"][key]["common_class_by_encoded_bin"], np.int64)
                        for r in rows if "common_class_by_encoded_bin" in r["returns"][key]), np.zeros((8, 5), np.int64))
                        for key in ("return_number", "number_of_returns")},
                    "warning": "Global ceiling is dominated by frequent classes; native survival and oracle mIoU are diagnostics, not learned model scores."})
    perturbation = []
    keys = sorted({(p["run"], p["domain"], p["split"]) for p in probes if p.get("available")})
    for run, domain, split in keys:
        selected = [p for p in probes if p.get("available") and (p["run"], p["domain"], p["split"]) == (run, domain, split)]
        for variant in sorted({v for p in selected for v in p["variants"]}):
            pairs = [p["variants"][variant] for p in selected]
            changed = sum((np.asarray(p["metrics"]["cm"], np.int64) for p in pairs), np.zeros((8, 8), np.int64))
            clean = sum((np.asarray(p["paired_clean_same_retained_points"]["cm"], np.int64) for p in pairs), np.zeros((8, 8), np.int64))
            if not np.array_equal(changed.sum(1), clean.sum(1)):
                raise RuntimeError("Perturbation comparison has different ground-truth supports")
            perturbation.append({"run": run, "domain": domain, "split": split, "variant": variant,
                "patches": len(selected), "changed": metrics(changed), "paired_clean": metrics(clean),
                "iou_delta": np.asarray(metrics(changed)["iou"]) - np.asarray(metrics(clean)["iou"]),
                "scope": "Possibly overlapping class-centred patches; not canonical scores, training ablations or independent samples."})
    training = []
    for replay in replays:
        selected = [r for r in replay["samples"] if "error" not in r]
        hist = sum((np.asarray(r["common_voxel_labels"], np.int64) for r in selected), np.zeros(8, np.int64))
        presence = np.asarray([r["common_voxel_labels"] for r in selected])
        training.append({"run": replay["run"], "selected_samples": len(selected),
            "sampled_selected_voxel_histogram": hist,
            "fraction_samples_containing_class": (presence > 0).mean(0) if len(selected) else None,
            "span_m_quantiles": np.quantile([r["span_m"] for r in selected], [.1, .5, .9], axis=0) if selected else None,
            "scope": replay["scope"]})
    feature = []
    for run in sorted({key[0] for key in records}):
        groups = {}
        for domain in ("eclair", "dales"):
            groups[domain] = {}
            for cls in range(1, 8):
                stats = [r["feature_moments"][str(cls)] for r in records.get((run, domain), [])
                         if r["feature_moments"][str(cls)]["n"]]
                if not stats:
                    continue
                n = sum(s["n"] for s in stats)
                mean = sum((np.asarray(s["sum"], float) for s in stats))/n
                var = np.maximum(sum((np.asarray(s["sum_sq"], float) for s in stats))/n - mean*mean, 0)
                groups[domain][cls] = (n, mean, var)
        for cls in range(1, 8):
            if cls not in groups["eclair"] or cls not in groups["dales"]:
                continue
            n1, m1, v1 = groups["eclair"][cls]
            n2, m2, v2 = groups["dales"][cls]
            feature.append({"run": run, "class": cls, "sampled_voxels_eclair": n1,
                "sampled_voxels_dales": n2,
                "standardized_centroid_shift_rms": float(np.sqrt(np.mean((m1-m2)**2/(.5*(v1+v2)+1e-6)))),
                "cosine_centroid_similarity": float(np.dot(m1, m2)/max(1e-12, np.linalg.norm(m1)*np.linalg.norm(m2))),
                "scope": "Within-checkpoint class-conditional descriptive shift. Neither domain invariance nor feature separability is established."})
    return {"dataset_geometry": geometry, "perturbation_summaries": perturbation,
            "training_exposure": training, "feature_domain_comparison": feature,
            "unavailable_probe_cohorts": [p for p in probes if not p.get("available")]}


def patch_geometry(xyz, gt, seed):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for the geometry block; use the existing project environment") from exc
    rng = np.random.default_rng(seed)
    query = []
    for cls in range(1, 8):
        ids = np.flatnonzero(gt == cls)
        if len(ids):
            query.extend(rng.choice(ids, min(64, len(ids)), replace=False).tolist())
    query = np.asarray(query, np.int64)
    tree = cKDTree(xyz)
    k = min(16, len(xyz))
    distances, neighbors = tree.query(xyz[query], k=k)
    distances, neighbors = np.atleast_2d(distances), np.atleast_2d(neighbors)
    result = {}
    for cls in range(1, 8):
        positions = np.flatnonzero(gt[query] == cls)
        records = []
        for i in positions:
            points = xyz[neighbors[i]]
            covariance = np.cov(points.T) if len(points) > 1 else np.zeros((3, 3))
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0)
            scale = max(float(values[-1]), 1e-12)
            records.append([float(distances[i, 1]) if k > 1 else 0., float(distances[i, -1]),
                (values[2]-values[1])/scale, (values[1]-values[0])/scale,
                values[0]/scale, 1-abs(vectors[2, 0]),
                float((gt[neighbors[i]] == cls).mean())])
        result[cls] = {"queries": len(records), "columns": ["nearest_neighbor_m", "kth_neighbor_m",
            "linearity", "planarity", "scattering", "normal_verticality", "neighbor_same_class_fraction"],
            "values": records, "k": k,
            "scope": "Sampled kNN geometry; density-dependent support, not object identity or a causal feature test."}
    return result


def bootstrap_delta(a, b, seed, repetitions=1000):
    rng = np.random.default_rng(seed)
    deltas, absent = [], []
    for _ in range(repetitions):
        ids = rng.integers(0, len(a), size=len(a))
        absent.append(a[ids].sum((0, 2)) == 0)
        deltas.append(np.asarray(metrics(b[ids].sum(0))["iou"]) - np.asarray(metrics(a[ids].sum(0))["iou"]))
    deltas = np.asarray(deltas)
    return {"class_delta_95pct": np.quantile(deltas, [.025, .975], axis=0),
            "miou_delta_95pct": np.quantile(deltas[:, 1:].mean(1), [.025, .975]),
            "fraction_replicates_without_gt_class": np.asarray(absent).mean(0),
            "repetitions": repetitions, "unit": "paired tile", "seed": seed}


def self_test():
    # Scientific invariants, including the rejection column and noncommuting label merges.
    cm = cm_from(np.array([7, 7, 0]), np.array([7, 0, 7]))
    assert metrics(cm)["recall"][7] == .5
    assert metrics(cm)["iou"][7] == .5
    assert cm.sum() == 2
    inv = np.array([0, 0, 0, 1, 1])
    y = np.array([0, 1, 1, 0, 1])
    assert np.array_equal(grouped_majority(inv, y, 2)[0], [1, 0])
    xyz = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.], [.6, 0., 0.]])
    common = np.array([4, 4, 1, 7])
    native = np.array([1, 2, 0, 3])
    vd = voxel_diagnostic(xyz, common, native, np.array([1, 4, 4, 7]), .5)
    assert vd["constant_voxel_accuracy_ceiling"] == .75
    assert vd["class_common_majority_voxels"][4] == 1
    assert vd["class_native_majority_then_map_voxels"][4] == 0
    assert vd["native_majority_then_map_backprojection"]["n_valid"] == 4
    assert np.array_equal(quantize_like_evaluator(np.array([[.49, 0., 0.], [.51, 0., 0.]]), .5)[:, 0], [0, 1])
    good = np.zeros((2, 8, 8), np.int64)
    good[:, 1, 1] = 10
    interval = bootstrap_delta(good, good, 1, repetitions=20)
    assert np.all(np.asarray(interval["miou_delta_95pct"]) == 0)
    assert remap_paths({"x": "/old/a"}, [("/old", "/new")]) == {"x": "/new/a"}
    assert remap_paths("/older/a", [("/old", "/new")]) == "/older/a"
    assert offline_audit(frozen())["all_miou_match"]
    print("PASS: metric rejection semantics, majority ties, native/common merge order, voxel scale, paired bootstrap, path remapping, all 16 canonical confusion matrices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--results-root", default="/scratch/m23csa510/e0_results")
    parser.add_argument("--out", default="project_diagnosis_20261017")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--overrides", help="JSON with optional run checkpoint/config and dataset config overrides")
    parser.add_argument("--voxel-edges", type=float, nargs="+", default=[.1, .2, .4, .5])
    parser.add_argument("--patch-m", type=float, default=50.)
    parser.add_argument("--training-items", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.patch_m <= 0 or any(x <= 0 for x in args.voxel_edges) or args.training_items < 1:
        parser.error("Positive patch size, voxel edges and training-items required")
    runner = Runner(args)
    try:
        runner.execute()
    except KeyboardInterrupt:
        runner.interrupted = True
        runner.log.warning("Interrupted; preserving completed stages for the same command")
    except Exception as exc:
        runner.failures.append({"stage": "orchestration", "error": str(exc), "traceback": traceback.format_exc()})
        runner.log.exception("Run could not complete; preserving all evidence")
    finally:
        runner.finish()
    status = json.loads((runner.out / "STATUS.json").read_text())["status"]
    return 130 if runner.interrupted else 2 if runner.failures else 3 if status in ("incomplete", "complete_with_reproduction_discrepancies") else 0


if __name__ == "__main__":
    raise SystemExit(main())
