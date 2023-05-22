#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ゼロ交差数を計算する関数
#

import numpy as np

# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
	
	zc = 0

	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	
	return zc

# 音声波形データを受け取り，ゼロ交差数を計算する関数
# 簡潔版
def zero_cross_short(waveform):
	
	d = np.array(waveform)
	return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

