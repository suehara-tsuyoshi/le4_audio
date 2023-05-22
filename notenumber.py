#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ノートナンバーと周波数の変換
#

import math

# ノートナンバーから周波数へ
def nn2hz(notenum):
	return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
	return int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69
