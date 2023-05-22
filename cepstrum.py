#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# ケプストラムを計算する関数
#

import numpy as np

# スペクトルを受け取り，ケプストラムを返す関数
def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum
