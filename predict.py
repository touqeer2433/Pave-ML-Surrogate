"""Flexible pavement strain and life predictor (ANN ensemble surrogate of 3D FE analysis).

Usage
-----
    from predict import Predictor
    p = Predictor()
    p.predict({"t_ACB": 120, "E_ACB": 2000, "P": 0.7}, design_repetitions=1e6)

Missing inputs take the database median. Units: thickness mm, modulus MPa,
pressure MPa, temperature degC. Strains are returned in microstrain
(positive = tension, negative = compression).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))


class Predictor:
    def __init__(self, folder=HERE):
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = torch.jit.load(os.path.join(folder, "pavement_ann.pt"), map_location="cpu").eval()
        self.inputs = self.meta["inputs"]

    def defaults(self):
        return dict(zip(self.inputs, self.meta["x_median"]))

    def ranges(self):
        return {k: (lo, hi) for k, lo, hi in zip(self.inputs, self.meta["x_min"], self.meta["x_max"])}

    def predict(self, designs, design_repetitions=None):
        df = pd.DataFrame([designs] if isinstance(designs, dict) else designs).copy()
        for c, v in self.defaults().items():
            if c not in df:
                df[c] = v
        X = df[self.inputs].to_numpy(float)
        m, c = self.meta, self.meta["coeff"]
        with torch.no_grad():
            mu, sd = self.model(torch.tensor(X, dtype=torch.float32))
        mu, sd = mu.numpy(), sd.numpy()
        S = 10 ** mu[:, :6]  # strain magnitudes, microstrain

        out = pd.DataFrame({s + "_microstrain": m["sign"][s] * S[:, k] for k, s in enumerate(m["strains"])})
        E = X[:, self.inputs.index("E_ACB")]
        # Asphalt Institute transfer functions
        out["Nf"] = c["k1f"] * (S[:, 1] * 1e-6) ** -c["k2f"] * (E * c["mpa_to_psi"]) ** -c["k3f"]
        out["Nr_fill"] = c["k1r"] * (S[:, 4] * 1e-6) ** -c["k2r"]
        out["Nr_subgrade"] = c["k1r"] * (S[:, 5] * 1e-6) ** -c["k2r"]
        out["Nf_direct"] = 10 ** mu[:, 6]
        nr = out[["Nr_fill", "Nr_subgrade"]].min(axis=1)
        out["N_allowable"] = np.minimum(out["Nf"], nr)
        out["governing_mode"] = np.where(out["Nf"] <= nr, "fatigue", "rutting")
        out["ensemble_spread_log10_et"] = sd[:, 1]
        out["ensemble_spread_log10_ev_subgrade"] = sd[:, 5]
        if design_repetitions:
            out["damage_ratio"] = design_repetitions / out["N_allowable"]
            out["adequate"] = out["damage_ratio"] <= 1
        lo, hi = np.array(m["x_min"]), np.array(m["x_max"])
        bad = (X < lo - 1e-9) | (X > hi + 1e-9)
        out["out_of_range_inputs"] = [", ".join(np.array(self.inputs)[b]) for b in bad]
        return out


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print(Predictor().predict({}, 1e6).T)
