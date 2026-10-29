# 原始数据与引用

来源：University of California, Irvine（加州大学尔湾分校，UCI）Machine Learning Repository。

- 官方页面及变量字典：https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- 永久链接：https://doi.org/10.24432/C5MC89
- 参考文献：Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository.
- 数据许可：Creative Commons Attribution 4.0 International（署名4.0国际许可）。

A自行下载官方压缩包，把其中的原始data.csv放在本目录，保持字节不变；原始文件按分号读取。下载来源、下载日期、文件版本与SHA-256（Secure Hash Algorithm 256-bit，256位安全散列文件校验值）写入data/manifest.json。不要把解压包内部的多层目录整套搬入仓库。

官方数据为4,424行、36个特征及Target标签；A须复核实际版本。Dropout是来源定义的项目/机构层面结局，可能包括转专业或转校，不直接等于永久离开高等教育。
