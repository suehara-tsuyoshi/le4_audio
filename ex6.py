#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，各フレーム毎のRMSを表示
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

def db(a):
  rms = np.sqrt((1/len(a)) * np.sum(a) * np.sum(a))
  dblist = []
  for i in range(0, len(a)):
    dblist.append(20 * np.log10(rms))
  return dblist

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('a.wav', sr=SR)

db_x = db(x)
t = np.arange(0, len(x)/SR, 1/SR)
plt.plot(t, db_x, label='signal')
plt.show()

	


