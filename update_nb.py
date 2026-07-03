import nbformat as nbf
nb = nbf.read('C:/Users/mcj/Desktop/item1/crime-analysis/notebooks/01_crime_analysis.ipynb', as_version=4)
nb.cells[2].source = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, glob
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("加载完成")"""
nb.cells[3].source = """DATA_DIR = "data/"
csv_files = glob.glob(DATA_DIR + "Chicago_Crimes_*.csv")
print(f"找到 {len(csv_files)} 个数据文件")
for f in csv_files:
    print(f"  - {f}")
try:
    chunks = []
    for f in csv_files:
        df_chunk = pd.read_csv(f, nrows=20000, low_memory=False)
        chunks.append(df_chunk)
        print(f"  加载 {f}: {len(df_chunk)} 行")
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n总样本: {len(df)} 行")
except FileNotFoundError:
    print("未找到数据文件！请检查 data/ 目录")"""
nbf.write(nb, 'C:/Users/mcj/Desktop/item1/crime-analysis/notebooks/01_crime_analysis.ipynb')
print("Notebook updated")
