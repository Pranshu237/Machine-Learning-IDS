import os

# The tests load XGBoost/LightGBM and PyTorch into one process. On macOS
# their separate OpenMP libraries can freeze or crash it unless OpenMP is
# limited to one thread. This must run before those libraries are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
