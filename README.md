```powershell
cd "D:\DEVOPS lab\lab#01"
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib joblib
python replicate.py
python make_summary.py
```