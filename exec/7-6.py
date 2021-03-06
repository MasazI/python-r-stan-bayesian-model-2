# 説明変数が多すぎる場合の回帰
# その場合、理想的には説明変数間の関係をモデリングして解決するのが良い。
# ここでの例では、遺伝子検査の結果をもとに、効きそうな薬を推定する場合について述べられている。
# 遺伝子間には多数の関係があり、その情報を活用することが望ましいそうだ。
# その他にも具体的なデータ例が記載されている。
# 1. 極端に出現データに偏りのある2値データは削除
# 2. 説明変数に対して階層的クラスタリングを実施して類似性の高い変数をまとたり、マイナーなデータを捨てる。
# 3. 主成分分析などの手法でデータの次元を減らす
# 4. Bayesian Lasso。説明変数の係数に二重指数分布を事前分布とした値を設定する。0に近い値をもつ係数は不要と考えることが可能。

