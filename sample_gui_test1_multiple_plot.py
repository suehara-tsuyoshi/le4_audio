#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込みスペクトログラム上に音量を表示する
#

# ライブラリの読み込み
import librosa
import numpy as np
import matplotlib.pyplot as plt

size_frame = 4096			# フレームサイズ
SR = 16000					# サンプリングレート
size_shift = 16000 / 100	# シフトサイズ = 0.001 秒 (10 msec)

# 音声ファイルを読み込む
x, _ = librosa.load('aiueo.wav', sr=SR)

# ファイルサイズ（秒）
duration = len(x) / SR

# ハミング窓
hamming_window = np.hamming(size_frame)

spectrogram = []	# スペクトログラムを保存するlist
volume = []			# 音量を保存するlist

# フレーム毎に処理
for i in np.arange(0, len(x)-size_frame, size_shift):
	
	# 該当フレームのデータを取得
	idx = int(i)	# arangeのインデクスはfloatなのでintに変換
	x_frame = x[idx : idx+size_frame]
	
	# スペクトル
	fft_spec = np.fft.rfft(x_frame * hamming_window)
	fft_log_abs_spec = np.log(np.abs(fft_spec))
	spectrogram.append(fft_log_abs_spec)

	# 音量
	vol = 20 * np.log10(np.mean(x_frame ** 2))
	volume.append(vol)

# 画像として保存するための設定
fig = plt.figure()

# まずはスペクトログラムを描画
ax1 = fig.add_subplot(111)
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')
ax1.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, duration, 0, 8000],
	aspect='auto',
	interpolation='nearest'
)

# 続いて右側のy軸を追加して，音量を重ねて描画
ax2 = ax1.twinx()
ax2.set_ylabel('volume [dB]')
x_data = np.linspace(0, duration, len(volume))
ax2.plot(x_data, volume, c='y')

plt.show()
fig.savefig('plot-spectogram-volume.png')

