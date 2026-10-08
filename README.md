# Flexible Pavement Strain & Life Predictor

Machine-learning surrogate of three-dimensional finite element (FE) analysis for the
layer-wise critical strains and the fatigue and rutting life of six-layer flexible pavements.

- **Model:** ensemble of 5 feed-forward neural networks (4 × 256, SiLU), trained on 3,465 3D FE analyses (43 inputs)
- **Outputs:** six critical strains, fatigue life *N*f, rutting lives *N*r (subgrade fill and natural subgrade), allowable repetitions, governing mode and damage ratio
- **Test accuracy:** R² = 0.94–0.99 for all strains

## Use online

| Version | Link |
|---|---|
| Streamlit web app | `https://<your-app-name>.streamlit.app` |
| Browser tool (no server, works offline) | `https://<username>.github.io/pave-ml-surrogate/` |

## Run locally

```bash
git clone https://github.com/<username>/pave-ml-surrogate.git
cd pave-ml-surrogate
pip install -r requirements.txt
streamlit run app.py
```

## Use from Python

```python
from predict import Predictor
p = Predictor()
res = p.predict({"t_ACB": 120, "E_ACB": 2000, "P": 0.7}, design_repetitions=1e6)
print(res.T)
```

Missing inputs take the database median. Batch prediction: pass a `pandas.DataFrame`
(see `examples/example_designs.csv`).

**Units:** thickness mm · modulus MPa · pressure MPa · temperature °C · strain microstrain (+ tension, − compression).

## Transfer functions (Asphalt Institute)

- Fatigue: *N*f = 0.0796 · εt^−3.291 · *E*ACB(psi)^−0.854
- Rutting: *N*r = 1.365 × 10⁻⁹ · εv^−4.477

The coefficients are stored in `model/meta.json` (`coeff`) and can be replaced with locally calibrated values without retraining.

## Validity

Predictions are reliable only inside the input ranges of the training database. Inputs outside the range are flagged by the app.

## Repository contents

```
app.py                  Streamlit web app
predict.py              Python predictor (single or batch)
model/pavement_ann.pt   Trained ANN ensemble (TorchScript)
model/meta.json         Input names, ranges, medians, transfer-function coefficients
docs/index.html         Stand-alone browser tool (GitHub Pages)
examples/               Example input file
requirements.txt        Python dependencies
```

## Citation

Rind, T. A., et al. *Machine learning surrogates of three-dimensional finite element analysis for
layer-wise critical strains and fatigue and rutting life of flexible pavements* (submitted).

## License

MIT, see `LICENSE`.
