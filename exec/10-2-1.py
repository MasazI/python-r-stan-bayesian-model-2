# 弱情報事前分布
# モデルが複雑なのにデータが少ない場合に収束しないことがある。
# 対策としては、シンプルなモデルから徐々に構築していくのが良いが、解析の目的を達成するために
# 複雑なモデルを使用する必要がある場合、パラメータの制約やデータへの制約を与えると良い。

# 弱情報事前分布の例1
# (-inf, inf) の範囲のパラメータには、分散が非常に大きな正規分布を通常使うが、
# パラメータが[a, b]の範囲に収まっていることがわかっている場合、
# 一様分布だと境界が切断される可能性があるため、平均(a+b)/2、標準偏差(b-a)/2 の正規分布を使うと良いらしい。

# 弱情報事前分布の例2
# 回帰係数
# 絶対値の大きな回帰係数を回避するため、回帰係数の大きさにペナルティを与える方法がある。
# 深層学習でも大きなパラメータにペナルティを課す正則化があるが、これは正規分布を事前情報分布として考える場合に相当する。
# ベイズモデルでは、Studentのt分布を事前分布として使う方法を推奨している。

# 弱情報事前分布の例3
# 正の値をもつパラメータ
# 背景知識からは考えられないような大きな値になってしまうような場合、弱情報事前分布を使う。
# ガンマ分布と半t分布の話。0に近いパラメータの場合は半t分布の方が安定する、ということ。
# Stanでの実装はlower=0で完了。

# 弱情報事前分布の例4
# 階層モデルの事前分布
# 具体例) 10-2-2.py

# その他の正の事前情報
# 半t分布、半正規分布、指数分布、ガンマ分布
# 具体例) 10-2-3.py
# 将棋のプロ棋士の勝敗データを使って、騎士の強さと勝負ムラを推定する。
# ウサギとカメの応用編かな。

# 確率など[0, 1]の範囲のパラメータ
# サイコロ、授業の出席、来店などの確率を用いたモデリングでは、
# 一様に等しく事象が起こる確率などは、ディリクレ分布を設定する選択肢もある。
# 選択肢が2つの場合はベータ分布を使う。
# 考え方としては、ベータ分布の多変量版がディリクレ分布ということになる。

# 分散共分散行列
# 具体例) 10-2-4.py
