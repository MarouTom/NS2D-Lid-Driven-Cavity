#!/usr/bin/env python3
"""Test the Re=400 three-grid extrapolation against an independent fourth grid."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
sys.path.insert(0, str(ROOT))
import numpy as np
from ns2d.accuracy import richardson_fit
from run_production import run_case
from run_accuracy import load_case


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--fft-workers",type=int,default=4)
    args=ap.parse_args()
    spec=dict(Re=400,N=513,T=150.0,frames=False)
    run_case("Re400_N513",spec,force=args.force,fft_workers=args.fft_workers)
    metas=[load_case(f"Re400_N{n}")[0] for n in (65,129,257,513)]
    h=np.array([r["h"] for r in metas])
    q=np.array([r["psi_min"] for r in metas])
    fit0=richardson_fit(h[:3],q[:3])
    fit1=richardson_fit(h[1:],q[1:])
    predicted=fit0["extrapolated"]+(q[2]-fit0["extrapolated"])*(h[3]/h[2])**fit0["order"]
    out=dict(Re=400,N=[65,129,257,513],h=h.tolist(),psi_min=q.tolist(),
             original_fit=fit0,finer_fit=fit1,predicted_fourth_value=float(predicted),
             fourth_prediction_error=float(abs(q[3]-predicted)),
             extrapolated_limit_change=abs(fit1["extrapolated"]-fit0["extrapolated"]),
             run_records=metas,
             analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             interpretation="independent fourth-grid check at Re=400 only; not proof for other Reynolds numbers or observables")
    (ROOT/"data/fourth_grid.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:out[k] for k in ('fourth_prediction_error','extrapolated_limit_change')}),flush=True)


if __name__=="__main__":
    main()
