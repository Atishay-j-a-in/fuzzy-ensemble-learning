# Fuzzy Ensemble for BACH Histology Classification

This project classifies breast-cancer histology images using five CNN backbones and a fuzzy ensemble. It supports binary classification and four-class classification on the ICIAR 2018 BACH dataset.

## Included

- Keras/TensorFlow notebooks for binary and four-class classification.
- Saved models under `best_saved_models/` for the `Macenko` and `Macenko_binary` pipelines.
- Fuzzy ensemble implementation in `Fuzzy Ensemble/FUZZY_ENSEMBLE.ipynb`.
- Prediction and confidence CSV files produced by earlier runs.
- `test_first_10.py` for local Xception inference on the supplied TIFF test images.

The five CNNs are VGG16, VGG19, Xception, InceptionV3, and InceptionResNetV2. The ensemble combines their predictions using information-theoretic and fuzzy/coalition calculations.

## Requirements

- Windows, Linux, or macOS.
- Python 3.11 recommended.
- 8 GB RAM for one-model inference; 16 GB recommended for several models.
- At least 10 GB free disk space for dependencies and model files.
- NVIDIA GPU is optional for inference and recommended for retraining.

The repository does not include a dependency manifest. Install the packages used by the notebooks with:

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow pyitlib jupyter ipykernel
```

## Windows Setup

From the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow pyitlib jupyter ipykernel
python -m ipykernel install --user --name fuzzybach-venv --display-name "Python (fuzzyBACH .venv)"
```

Select `Python (fuzzyBACH .venv)` as the notebook kernel in VS Code or Jupyter.

## Run the Local Launcher

The Windows launcher starts `run_app.py`:

```powershell
.\start_fuzzybach.bat
```

You can also start it directly:

```powershell
.\.venv\Scripts\python.exe run_app.py
```

## Test Images

The included test dataset contains whole-slide `.svs` files and extracted TIFF/PNG images. The inference script uses the first ten-image batch selected by its current configuration and runs the four-class Xception model:

```powershell
.\.venv\Scripts\python.exe test_first_10.py
```

The script expects images under `ICIAR2018_BACH_Challenge_TestDataset/Photos/`, resizes them to `512x512`, and uses:

```text
best_saved_models/Macenko/Xception/xception_upto95frozen.h5
```

Predictions are saved as a CSV in the repository root. The model predicts:

1. Benign
2. In-situ carcinoma
3. Invasive carcinoma
4. Normal tissue

The `.svs` slides are not passed directly to the classifier. They require a whole-slide reader and patch extraction. Use the extracted TIFF/PNG files for the supplied script.

## Run the Notebooks

Start Jupyter with:

```powershell
jupyter notebook
```

Then open one of the notebooks in `Two class/`, `Four class/`, or `Fuzzy Ensemble/`. Select the project virtual-environment kernel before execution.

The original notebooks contain Google Colab and Google Drive paths such as `/content/drive/...`. Update those paths for local execution. The fuzzy ensemble notebook can use the included CSV prediction and confidence matrices without the original image dataset.

## Dataset and Retraining

Download the BACH dataset from the [ICIAR 2018 dataset page](https://iciar2018-challenge.grand-challenge.org/Dataset/). The training notebooks expect `train/` and `test/` directories with class subdirectories and RGB images resized to `512x512`.

Retraining all five CNNs is computationally expensive. A CUDA-compatible NVIDIA GPU and at least 16 GB RAM are recommended. CPU-only inference is possible, but slower.

## Compatibility Notes

The notebooks use older Keras APIs such as `predict_generator` and `Adam(lr=...)`. Current TensorFlow/Keras releases may require small API updates. Some legacy HDF5 models contain Lambda layers and may require trusted unsafe deserialization when loaded.

Model outputs are research results and must not be used as a medical diagnosis.

## Reference

```text
Bhowal, P., Sen, S., Silva, J. D. V., and Sarkar, R.
Fuzzy ensemble of deep learning models using choquet fuzzy integral,
coalition game and information theory for breast cancer histology classification.
Expert Systems with Applications, vol. 190, 116167, 2022. (Published online 2021.)
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
