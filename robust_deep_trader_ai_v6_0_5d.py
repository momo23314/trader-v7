"""
RobustDeepTraderAI v6.0-5d
===========================
v5.5-5d からの 6大改善:

【v6 改善①】スリッページ込みラベル（実戦乖離の縮小）
    旧: label = (close_t5 > open_t1 * 1.01)  完全約定仮定
    新: entry_price = open_t1 * (1 + SLIPPAGE=0.15%)  ← 成行き買いの不利
        exit_price  = close_t5 * (1 - SLIPPAGE)       ← 成行き売りの不利
        label = (exit_price > entry_price * 1.01)
    → バックテストと実戦の乖離を縮小。高勝率でも実際に負けるシグナルを排除。

【v6 改善②】曜日特徴量の質的改善（one-hot + カレンダーアノマリー）
    旧: DayOfWeek = dayofweek / 4.0  線形、非線形関係を捉えられない
    新: Is_Monday / Is_Friday / Is_Month_End / Is_Month_Start / Days_To_SQ（5個）
        + Weekly_Gap / Weekly_ATR_Ratio（維持）
    → 月末フロー・SQ（第2金曜）・月曜効果など決定論的圧力を明示的に捉える

【v6 改善③】Weekly_Trend特徴量（週足4週SMA比較）
    追加特徴量: Weekly_Trend（週の終値 > 4週SMAかどうか）
    → 日次シグナルが週足トレンドと順方向か逆方向かをモデルが自己判断できる

【v6 改善④】Friday_SellPressure特徴量（金曜大引け売り圧力）
    追加特徴量: Friday_SellPressure（金曜の出来高 / 20日平均出来高）
    → エグジット時の売り圧力を事前予測する
    特徴量合計: 26 → 32個

【v6 改善⑤】デュアルヘッドモデル（分類 + リターン回帰）
    旧: sigmoid出力1個（上昇確率のみ）
    新: prob_output（sigmoid, Focal Loss） + ret_output（linear, Huber Loss）
        合計損失 = Focal + LAMBDA_HUBER(=0.3) * Huber
    → 期待リターンの大きさも学習。morning_predict で期待リターン付き表示を実現。

【v6 改善⑥】3シードアンサンブル（過学習の分散）
    --seed 0/1/2 で個別訓練 → model_v6_seedN_best.keras に保存
    morning_predict: 利用可能な全シードモデルを自動検出・確率平均アンサンブル
    → 単一モデルの「盲点」を相殺。予測の安定性向上。

【v6 改善⑦】ケリー基準ポジションサイジング（資金効率の最大化）
    evening_evaluate で avg_win/avg_loss をkelly_stats_v6.jsonに蓄積
    morning_predict でハーフケリー配分率を表示（最大30%キャップ）
    kelly_frac = (b*p - q) / b * 0.5  (b=損益比, p=モデル確率, q=1-p)

使い方:
  python robust_deep_trader_ai_v6_0_5d.py preprocess
  python robust_deep_trader_ai_v6_0_5d.py train_kaggle --batch 256 --epochs 100 --seed 0
  python robust_deep_trader_ai_v6_0_5d.py train_kaggle --batch 256 --epochs 100 --seed 1
  python robust_deep_trader_ai_v6_0_5d.py train_kaggle --batch 256 --epochs 100 --seed 2
  python robust_deep_trader_ai_v6_0_5d.py morning [--top_n 10]
  python robust_deep_trader_ai_v6_0_5d.py evening
"""

import gc
import os
import glob
import time
import json
import argparse
import random
from datetime import datetime, timedelta

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm', '-q'])
    from tqdm import tqdm

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import ta
import warnings
import sys

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input,
        BatchNormalization, Bidirectional, GlobalAveragePooling1D,
        MultiHeadAttention, LayerNormalization, Add, Activation
    )
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import Callback
except ImportError:
    print("\n[エラー] TensorFlowが見つかりません。pip install tensorflow\n")
    sys.exit()

# =================================================================
# 基本設定
# =================================================================
DATA_DIR          = './data'
GLOBAL_LIST_FILE  = './all_stocks_global.csv'
JPX400_LIST_FILE  = './jpx400.csv'

PREDICTION_RECORD = './latest_predictions_5d.pkl'
SCALER_FILE       = './scaler_v6_5d.pkl'
CLASS_WEIGHT_FILE = './class_weight_v6_5d.json'
MACRO_CACHE_FILE  = './data/macro_cache.pkl'
CHUNK_DIR         = './processed_chunks'
CHUNK_PREFIX      = 'chunk_'
MAX_CHUNKS        = 1000

HORIZON_SHIFT     = 5

# [v6 改善①] スリッページ設定（両側）
SLIPPAGE          = 0.0015   # 0.15% 成行き約定コスト

# Walk-forward validation
# [Fix⑨] TRAIN_CUTOFF_DATE を動的計算（現在日 - 3年）
# 固定値 '2022-12-31' だと 2026年現在、3年分がValになり訓練データが少なくなる
_cutoff_dt        = datetime.now() - timedelta(days=3 * 365)
TRAIN_CUTOFF_DATE = _cutoff_dt.strftime('%Y-%m-%d')
VAL_START_DATE    = (_cutoff_dt + timedelta(days=1)).strftime('%Y-%m-%d')

# 利益閾値（スリッページ調整済みエントリー/エグジット後の純リターン閾値）
PROFIT_THRESHOLD  = 0.010    # 1.0%

# [v6 改善⑤] デュアルヘッド: 回帰ヘッドの損失重み
LAMBDA_HUBER      = 0.30
HUBER_DELTA       = 0.03     # Huber損失の折れ目（3%リターン）

# Focal Loss パラメータ
FOCAL_GAMMA       = 2.0
FOCAL_ALPHA       = 0.75

# [v6 改善⑥] 3シードアンサンブル
N_ENSEMBLE        = 3
MODEL_SEEDS_BEST   = [f'./model_v6_seed{i}_best.keras'   for i in range(N_ENSEMBLE)]
MODEL_SEEDS_LATEST = [f'./model_v6_seed{i}_latest.keras' for i in range(N_ENSEMBLE)]

# 進捗ファイルテンプレート（シード別）
PROGRESS_FILE_TPL = './training_progress_v6_seed{}.json'

# [v6 改善⑦] ケリー統計ファイル
KELLY_STATS_FILE  = './kelly_stats_v6.json'

MACRO_TICKERS = {
    'nikkei' : '^N225',
    'usdjpy' : 'JPY=X',
    'vix'    : '^VIX',
}

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
# [Fix9] Train/Val 時系列分割用サブディレクトリ
CHUNK_TRAIN_DIR = os.path.join(CHUNK_DIR, "train")
CHUNK_VAL_DIR   = os.path.join(CHUNK_DIR, "val")
os.makedirs(CHUNK_TRAIN_DIR, exist_ok=True)
os.makedirs(CHUNK_VAL_DIR,   exist_ok=True)


# =================================================================
# 海外株式ビルトインリスト（all_stocks_global.csv がない場合のフォールバック）
# =================================================================
FTSE100_TICKERS = [
    'AZN.L','HSBA.L','SHEL.L','ULVR.L','BP.L','RIO.L','GSK.L','DGE.L',
    'REL.L','NG.L','BATS.L','LSEG.L','NWG.L','LLOY.L','BT-A.L','VOD.L',
    'AAL.L','CPG.L','BARC.L','IMB.L','RKT.L','PRU.L','STAN.L','TSCO.L',
    'SGE.L','ABF.L','AHT.L','AUTO.L','AV.L','BA.L','BKG.L','BNZL.L',
    'BRBY.L','CNA.L','CRDA.L','EDV.L','EXPN.L','FERG.L','FLTR.L','GLEN.L',
    'HLN.L','HLMA.L','IHG.L','ITRK.L','JD.L','KGF.L','LAND.L','MKS.L',
    'MNDI.L','MTO.L','PHNX.L','PSH.L','PSN.L','PSON.L','RMV.L','RR.L',
    'SBRY.L','SDR.L','SGRO.L','SKG.L','SMDS.L','SMIN.L','SMT.L','SN.L',
    'SSE.L','SVT.L','TATE.L','TW.L','WEIR.L','WPP.L','WTB.L',
]
DAX40_TICKERS = [
    'ADS.DE','AIR.DE','ALV.DE','BAS.DE','BAYN.DE','BEI.DE','BMW.DE',
    'BNR.DE','CON.DE','1COV.DE','DHER.DE','DTE.DE','EOAN.DE','FRE.DE',
    'FME.DE','HEI.DE','HEN3.DE','HFG.DE','IFX.DE','MBG.DE','MRK.DE',
    'MTX.DE','MUV2.DE','P911.DE','PAH3.DE','PUM.DE','QIA.DE','RHM.DE',
    'RWE.DE','SAP.DE','SHL.DE','SIE.DE','SRT3.DE','SY1.DE','VNA.DE',
    'VOW3.DE','ZAL.DE','DB1.DE','DBK.DE',
]
ASX200_TICKERS = [
    'BHP.AX','CBA.AX','CSL.AX','ANZ.AX','WBC.AX','NAB.AX','WES.AX',
    'MQG.AX','WOW.AX','RIO.AX','TLS.AX','FMG.AX','GMG.AX','TCL.AX',
    'STO.AX','ALL.AX','APA.AX','BXB.AX','COL.AX','CPU.AX','DOW.AX',
    'DXS.AX','IAG.AX','JHX.AX','MPL.AX','ORG.AX','QBE.AX','REA.AX',
    'RHC.AX','SCG.AX','SEK.AX','SGP.AX','SHL.AX','SUN.AX','TAH.AX',
    'TPG.AX','TWE.AX','VCX.AX','WPL.AX','XRO.AX','AGL.AX','AMP.AX',
    'BOQ.AX','BSL.AX','CAR.AX','CGF.AX','EVN.AX','GNC.AX','HVN.AX',
    'ILU.AX','IPL.AX','JBH.AX','LLC.AX','MIN.AX','MND.AX','NEC.AX',
    'NST.AX','NUF.AX','ORA.AX','PMV.AX','PPT.AX','RWC.AX','S32.AX',
    'SFR.AX','SUL.AX',
]
TSX60_TICKERS = [
    'AC.TO','AEM.TO','ATD.TO','BAM.TO','BCE.TO','BNS.TO','CAE.TO',
    'CCO.TO','CM.TO','CNQ.TO','CNR.TO','CP.TO','CTC-A.TO','CVE.TO',
    'DOL.TO','EMA.TO','ENB.TO','FFH.TO','FNV.TO','FTS.TO','GIB-A.TO',
    'GWO.TO','IAG.TO','IFC.TO','K.TO','KL.TO','L.TO','MFC.TO',
    'MRU.TO','NTR.TO','POW.TO','PPL.TO','RCI-B.TO','REI-UN.TO','RY.TO',
    'SJR-B.TO','SLF.TO','SU.TO','T.TO','TD.TO','TIH.TO','TOU.TO',
    'TRP.TO','WCN.TO','WFG.TO',
]
KOSPI_TICKERS = [
    '005930.KS','000660.KS','035420.KS','005380.KS','051910.KS','006400.KS',
    '035720.KS','028260.KS','066570.KS','105560.KS','055550.KS','032830.KS',
    '096770.KS','003670.KS','012330.KS','018260.KS','086790.KS','009540.KS',
    '010950.KS','011170.KS','034730.KS','000270.KS','017670.KS','015760.KS',
    '030200.KS','009150.KS','002380.KS','042660.KS','000810.KS','011790.KS',
    '047050.KS','316140.KS','010130.KS','001040.KS','016360.KS','011780.KS',
    '000100.KS','097950.KS','068270.KS','251270.KS','001120.KS','033780.KS',
    '090430.KS','085660.KS','078930.KS','003490.KS','010140.KS','020150.KS',
    '000720.KS','024110.KS','326030.KS','004990.KS','007070.KS','006360.KS',
    '000150.KS','003410.KS','011200.KS','005490.KS',
]
TWSE_TICKERS = [
    '2330.TW','2317.TW','2454.TW','2308.TW','2303.TW','2882.TW','2881.TW',
    '2886.TW','2891.TW','2884.TW','1301.TW','1303.TW','2412.TW','2002.TW',
    '1216.TW','2207.TW','2357.TW','3711.TW','2379.TW','2395.TW','2408.TW',
    '3034.TW','6505.TW','2327.TW','2382.TW','2301.TW','3008.TW','2344.TW',
    '2353.TW','2887.TW','2885.TW','5880.TW','2892.TW','2883.TW','2880.TW',
    '2890.TW','1102.TW','1101.TW','2105.TW','2352.TW','2337.TW','3045.TW',
    '4938.TW','2474.TW','2376.TW','2356.TW','2388.TW','2347.TW','3533.TW',
    '2385.TW','2325.TW','2371.TW','2360.TW','2368.TW','2345.TW','2049.TW',
    '1605.TW','2609.TW','2615.TW','2603.TW','2610.TW','2618.TW','5871.TW',
]


# =================================================================
# ティッカー読み込み
# =================================================================
def read_tickers_from_file(filepath):
    if not os.path.exists(filepath):
        return []
    tickers = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                for t in line.split(','):
                    t = t.strip()
                    if t:
                        tickers.append(t)
    except Exception as e:
        print(f"[WARN] ファイル読み込みエラー ({filepath}): {e}")
    return list(dict.fromkeys(tickers))


def get_global_train_stocks():
    tickers = read_tickers_from_file(GLOBAL_LIST_FILE)
    if tickers:
        print(f"[INFO] {GLOBAL_LIST_FILE} から {len(tickers)} 銘柄を読み込みました")
        return tickers
    print("[INFO] all_stocks_global.csv が見つかりません。ビルトインリストを使用します。")
    return (
        FTSE100_TICKERS + DAX40_TICKERS + ASX200_TICKERS
        + TSX60_TICKERS + KOSPI_TICKERS + TWSE_TICKERS
    )


TRAIN_STOCKS   = get_global_train_stocks()
PREDICT_STOCKS = read_tickers_from_file(JPX400_LIST_FILE) or ['7203.T', '8306.T', '6758.T']
print(f"[INFO] 学習対象: {len(TRAIN_STOCKS)} 銘柄  /  予測対象: {len(PREDICT_STOCKS)} 銘柄")


# =================================================================
# CSV読み込み
# =================================================================
def read_stock_csv(filepath):
    try:
        probe = pd.read_csv(filepath, nrows=3, header=0, index_col=0)
        first_idx = str(probe.index[0]).strip().lower()
        is_new_format = first_idx in ('ticker', 'price', 'nan', '')
        if is_new_format:
            df = pd.read_csv(filepath, header=0, index_col=0,
                             skiprows=[1, 2], parse_dates=True)
        else:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()].copy()
        df.index.name = 'Date'
        df.index = df.index.tz_localize(None)
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        df = df.drop(columns=['Price'], errors='ignore')
        required = ['Close', 'High', 'Low', 'Open', 'Volume']
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=required)
    except Exception as e:
        print(f"[WARN] CSV read error ({filepath}): {e}")
        return pd.DataFrame()


# =================================================================
# ラベル計算ユーティリティ（スリッページ込み）[v6 改善①]
# =================================================================
def compute_labels_and_returns(next_open, next_close):
    """
    スリッページを考慮したエントリー/エグジット価格でラベルとリターンを計算。
    next_open  : shift(-1) した翌日の始値配列
    next_close : shift(-HORIZON_SHIFT) した5日後の終値配列
    returns:
        target_vals : 0/1 分類ラベル（float32）、NaN/無効は np.nan のまま返す
        w_vals      : sample_weight（リターン絶対値に比例、0.3-5.0クリップ）
        ret_clipped : ±15%クリップした実リターン（Huber回帰ヘッド用）

    [Fix②] NaNサイレント変換バグ修正:
    旧: NaN比較は常にFalse → ラベルが 0.0（陰性）に化ける
    新: valid_mask で NaN/ゼロ価格を検出 → 無効サンプルは np.nan を返す
        呼び出し側の np.isnan(tgt) チェックで正しく除外される
    """
    entry_price  = next_open  * (1.0 + SLIPPAGE)
    exit_price   = next_close * (1.0 - SLIPPAGE)

    # 有効サンプル判定: open/close が NaN でなく、entry_price が正の値
    valid_mask   = (
        ~np.isnan(next_open) &
        ~np.isnan(next_close) &
        (entry_price > 0)
    )
    # 無効位置は np.nan（呼び出し側で除外される）
    target_vals  = np.where(
        valid_mask,
        (exit_price > entry_price * (1.0 + PROFIT_THRESHOLD)).astype(np.float32),
        np.nan
    ).astype(np.float32)

    raw_ret      = np.where(
        valid_mask & (entry_price > 0),
        (exit_price - entry_price) / (entry_price + 1e-9),
        0.0
    ).astype(np.float32)
    w_vals       = np.clip(np.abs(raw_ret) * 10.0, 0.3, 5.0).astype(np.float32)
    ret_clipped  = np.clip(raw_ret, -0.15, 0.15).astype(np.float32)
    return target_vals, w_vals, ret_clipped


# =================================================================
# 進捗管理ユーティリティ
# =================================================================
def load_progress(file=None):
    target = file or PROGRESS_FILE_TPL.format(0)
    if os.path.exists(target):
        try:
            with open(target, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "completed_epochs" : 0,
        "best_val_auc"     : 0.0,
        "best_val_accuracy": 0.0,
        "last_updated"     : "",
        "history"          : []
    }


def save_progress(progress, file=None):
    target = file or PROGRESS_FILE_TPL.format(0)
    progress["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(target, 'w') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# =================================================================
# カスタムコールバック（デュアルヘッド対応）
# =================================================================
class TrainingProgressCallback(Callback):
    """
    毎エポック終了後に:
    1. progress_file を更新
    2. model_latest_path を必ず保存
    3. val_auc が過去最高なら model_best_path も保存
    デュアルヘッド / シングルヘッド 両方のメトリクス名に対応。
    """
    def __init__(self, initial_epoch=0, total_epochs=100, steps_per_epoch=0,
                 model_best_path=None, model_latest_path=None, progress_file=None):
        super().__init__()
        self.model_best_path   = model_best_path   or MODEL_SEEDS_BEST[0]
        self.model_latest_path = model_latest_path or MODEL_SEEDS_LATEST[0]
        self.progress_file     = progress_file     or PROGRESS_FILE_TPL.format(0)
        self.progress          = load_progress(self.progress_file)
        self.initial_epoch     = initial_epoch
        self.total_epochs      = total_epochs
        self.steps_per_epoch   = steps_per_epoch
        self._pbar             = None

    def on_epoch_begin(self, epoch, logs=None):
        # Keras は model.fit(initial_epoch=N) を渡すと
        # コールバックの epoch 引数に N から始まる絶対インデックスを渡してくる。
        # self.initial_epoch を足すと二重カウントになるため epoch+1 のみ使用。
        abs_epoch = epoch + 1
        desc = (f"Epoch {abs_epoch:3d}/{self.total_epochs} "
                f"[best={self.progress['best_val_auc']:.4f}]")
        self._pbar = tqdm(
            total=self.steps_per_epoch, desc=desc, unit='step',
            ncols=120, leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )

    def on_train_batch_end(self, batch, logs=None):
        if self._pbar is not None:
            logs = logs or {}
            # デュアル/シングル どちらのキー名でも対応
            loss_val = logs.get('loss', 0)
            auc_val  = (logs.get('prob_output_auc') or logs.get('auc') or 0)
            self._pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'auc' : f"{auc_val:.4f}",
            })
            self._pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        logs = logs or {}
        # Keras が渡す epoch は initial_epoch からの絶対インデックス。
        # self.initial_epoch を加算すると二重カウントになるため epoch+1 のみ。
        abs_epoch = epoch + 1

        # 進捗ファイルから最新の best_val_auc を再読み込み
        latest_prog = load_progress(self.progress_file)
        if latest_prog['best_val_auc'] > self.progress['best_val_auc']:
            self.progress['best_val_auc'] = latest_prog['best_val_auc']

        # デュアルヘッド / シングルヘッド 両対応のメトリクス取得
        def _get(keys, default=0.0):
            for k in keys:
                v = logs.get(k)
                if v is not None and v != 0.0:
                    return float(v)
            return default

        val_auc    = _get(['val_prob_output_auc', 'val_auc'])
        val_acc    = _get(['val_prob_output_accuracy', 'val_accuracy'])
        val_prec   = _get(['val_prob_output_precision',
                           'val_precision', 'val_precision_40', 'val_precision_30'])
        train_auc  = _get(['prob_output_auc', 'auc'])
        train_acc  = _get(['prob_output_accuracy', 'accuracy'])
        train_loss = float(logs.get('loss', 0.0))

        # 最新モデルを必ず保存
        try:
            self.model.save(self.model_latest_path)
        except Exception as e:
            print(f"[WARN] latest保存失敗: {e}")

        # ベストモデルを条件付き保存
        is_best = val_auc > self.progress["best_val_auc"]
        if is_best:
            try:
                self.model.save(self.model_best_path)
                self.progress["best_val_auc"]      = val_auc
                self.progress["best_val_accuracy"] = val_acc
            except Exception as e:
                print(f"[WARN] best保存失敗: {e}")

        self.progress["completed_epochs"] = abs_epoch
        self.progress["history"].append({
            "epoch"        : abs_epoch,
            "train_loss"   : round(train_loss, 5),
            "train_auc"    : round(train_auc,  5),
            "train_acc"    : round(train_acc,  5),
            "val_auc"      : round(val_auc,    5),
            "val_accuracy" : round(val_acc,    5),
            "val_precision": round(val_prec,   5),
            "is_best"      : is_best,
        })
        save_progress(self.progress, self.progress_file)

        best_mark = " ★BEST" if is_best else ""
        tqdm.write(
            f"  ✔ Epoch {abs_epoch:3d} │ "
            f"loss={train_loss:.4f}  auc={train_auc:.4f}  "
            f"val_auc={val_auc:.4f}  val_acc={val_acc:.4f}  "
            f"val_prec={val_prec:.4f}{best_mark}"
        )


# =================================================================
# データジェネレータ（デュアルヘッド + スリッページ対応）
# =================================================================
class StockDataGenerator(tf.keras.utils.Sequence):
    """
    [Fix 8] ウォークフォワード対応: date_cutoff でサンプルを時間フィルタリング
    [v6 改善①] スリッページ込みラベル
    [v6 改善⑤] デュアルヘッド: y を dict で返す
    """
    def __init__(self, file_paths, ai_instance, seq_len,
                 batch_size=512, is_training=True, date_cutoff=None):
        self.file_paths  = file_paths
        self.ai          = ai_instance
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.is_training = is_training
        self.date_cutoff = pd.Timestamp(date_cutoff) if date_cutoff else None
        self._file_cache = {}
        self.samples     = self._build_index()
        self.indices     = np.arange(len(self.samples))
        if self.is_training:
            np.random.shuffle(self.indices)

    def _build_index(self):
        # [Fix④] 生CSVではなく _create_features 後の長さでインデックスを構築
        # 旧: len(df)（生CSV）で構築 → __getitem__では dropna後のdf を使うため長さが食い違う
        # 新: _create_features を実行してfeat_df の長さでサンプルを登録
        samples = []
        for file in self.file_paths:
            try:
                df = read_stock_csv(file)
                if df.empty:
                    continue
                feat_df = self.ai._create_features(df)
                if feat_df.empty:
                    continue
                max_start = len(feat_df) - self.seq_len - HORIZON_SHIFT - 1
                if max_start <= 0:
                    continue
                for i in range(0, max_start, 3):
                    end_idx  = i + self.seq_len
                    if end_idx >= len(feat_df):
                        break
                    end_date = feat_df.index[end_idx]
                    if self.date_cutoff is not None:
                        if self.is_training and end_date > self.date_cutoff:
                            continue
                        if not self.is_training and end_date <= self.date_cutoff:
                            continue
                    samples.append((file, i))
            except Exception as e:
                print(f"[WARN] Index build error ({file}): {e}")
        return samples

    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]
        X_batch, y_batch, w_batch, ret_batch = [], [], [], []

        for file, start_idx in batch_samples:
            if file not in self._file_cache:
                # [Fix5] キャッシュ上限: 200銘柄超でLRU除去（OOM防止）
                if len(self._file_cache) >= 200:
                    # 最も古いエントリを20件削除
                    old_keys = list(self._file_cache.keys())[:20]
                    for k in old_keys:
                        del self._file_cache[k]
                try:
                    df = read_stock_csv(file)
                    if df.empty:
                        self._file_cache[file] = (None, None, None, None)
                    else:
                        f_df = self.ai._create_features(df)
                        if not f_df.empty:
                            feats_scaled = self.ai.scaler.transform(
                                f_df[self.ai.feature_cols].values
                            )
                            next_open  = f_df['Open'].shift(-1).values
                            next_close = f_df['Close'].shift(-HORIZON_SHIFT).values
                            tgt, w_v, ret_c = compute_labels_and_returns(next_open, next_close)
                            self._file_cache[file] = (feats_scaled, tgt, w_v, ret_c)
                        else:
                            self._file_cache[file] = (None, None, None, None)
                except Exception as e:
                    print(f"[WARN] Cache load error ({file}): {e}")
                    self._file_cache[file] = (None, None, None, None)

            feats, targets, w_vals, ret_clips = self._file_cache[file]
            end_idx = start_idx + self.seq_len
            if feats is not None and end_idx < len(feats):
                target_val = targets[end_idx - 1]
                if not np.isnan(target_val):
                    X_batch.append(feats[start_idx:end_idx])
                    y_batch.append(float(target_val))
                    w_batch.append(float(w_vals[end_idx - 1]))
                    ret_batch.append(float(ret_clips[end_idx - 1]))

        n_feats = len(self.ai.feature_cols)
        if len(X_batch) == 0:
            # ダミーバッチ（ゼロ埋め）
            return (
                np.zeros((1, self.seq_len, n_feats), dtype=np.float32),
                {
                    'prob_output': np.zeros((1,), dtype=np.float32),
                    'ret_output' : np.zeros((1,), dtype=np.float32),
                },
                np.ones((1,), dtype=np.float32)
            )

        # [v6 改善⑤] デュアルヘッド用 dict 出力
        return (
            np.array(X_batch,   dtype=np.float32),
            {
                'prob_output': np.array(y_batch,   dtype=np.float32),
                'ret_output' : np.array(ret_batch, dtype=np.float32),
            },
            np.array(w_batch, dtype=np.float32)
        )


# =================================================================
# メインクラス
# =================================================================
class RobustDeepTraderAI:

    def __init__(self, seq_len=120, epochs=30, batch_size=512, seed=0):
        self.seq_len       = seq_len
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.seed          = seed

        # シード別モデルパス
        self.model_best    = MODEL_SEEDS_BEST[seed]   if seed < N_ENSEMBLE else MODEL_SEEDS_BEST[0]
        self.model_latest  = MODEL_SEEDS_LATEST[seed] if seed < N_ENSEMBLE else MODEL_SEEDS_LATEST[0]
        self.progress_file = PROGRESS_FILE_TPL.format(seed)

        # [v6 改善②③④] 特徴量（32個）
        # DayOfWeek を廃止し one-hot + カレンダー + Weekly_Trend + Friday_SellPressure を追加
        self.feature_cols = [
            # 価格モメンタム・テクニカル（15個）
            'Ret_1d', 'Ret_5d', 'SMA_20_Div', 'RSI_14', 'MACD_Hist',
            'Volatility', 'ATR_Pct', 'BB_Width', 'Vol_Zscore', 'Ret_20d',
            'Gap', 'OBV_Norm', 'Stoch_K', 'ADX_14', 'CMF_20',
            # 中期モメンタム（3個）
            'Ret_10d', 'SMA_50_Div', 'ROC_5',
            # [v6 改善②] 週次サイクル one-hot + カレンダー（7個）
            # DayOfWeek 廃止 → Is_Monday/Is_Friday/Is_Month_End/Is_Month_Start/Days_To_SQ
            'Is_Monday', 'Is_Friday', 'Is_Month_End', 'Is_Month_Start', 'Days_To_SQ',
            'Weekly_Gap', 'Weekly_ATR_Ratio',
            # [v6 改善③] 週足トレンド（1個）
            'Weekly_Trend',
            # [v6 改善④] 金曜売り圧力（1個）
            'Friday_SellPressure',
            # グローバルマクロ（4個）
            'Nikkei_Ret5d', 'USDJPY_Ret5d', 'VIX_Level', 'Nikkei_Vol20',
            # ボラティリティレジーム（1個）
            'Vol_Regime',
        ]
        # 合計 32個

        self.scaler     = RobustScaler()
        self.is_trained = False
        self.macro_df   = self._load_macro_data()

        # class_weight ロード
        self.class_weight_ratio = None
        if os.path.exists(CLASS_WEIGHT_FILE):
            try:
                with open(CLASS_WEIGHT_FILE, 'r') as f:
                    cw = json.load(f)
                self.class_weight_ratio = {int(k): float(v) for k, v in cw.items()}
                print(f"▶ class_weight ロード: {self.class_weight_ratio}")
            except Exception as e:
                print(f"[WARN] class_weight ロード失敗: {e}")

        mixed_precision.set_global_policy('mixed_float16')
        print("▶ 混合精度学習(float16)を有効化しました")

        # MirroredStrategy
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            self.strategy      = tf.distribute.MirroredStrategy()
            self.effective_batch = self.batch_size * len(gpus)
            print(f"▶ MirroredStrategy: GPU {len(gpus)}台で分散学習")
        else:
            self.strategy      = tf.distribute.get_strategy()
            self.effective_batch = self.batch_size
            print(f"▶ シングルデバイス学習 (GPU: {len(gpus)}台)")

        print(f"  バッチサイズ: {self.batch_size} × {max(len(gpus),1)}GPU"
              f" = 実効 {self.effective_batch}")
        print(f"  シード: {self.seed}  モデル保存先: {self.model_best}")

        self.model = self._load_or_build_model()

    # ------------------------------------------------------------------
    # マクロデータのロード・キャッシュ
    # ------------------------------------------------------------------
    def _load_macro_data(self):
        if os.path.exists(MACRO_CACHE_FILE):
            try:
                cache_age_days = (datetime.now().timestamp()
                                  - os.path.getmtime(MACRO_CACHE_FILE)) / 86400
                if cache_age_days <= 7:
                    macro_df = joblib.load(MACRO_CACHE_FILE)
                    print(f"▶ マクロデータ キャッシュロード: {len(macro_df)} 日分")
                    return macro_df
                print(f"▶ マクロキャッシュが {cache_age_days:.0f}日前のため再ダウンロードします...")
            except Exception as e:
                print(f"[WARN] マクロキャッシュ読み込み失敗: {e} → 再ダウンロード")

        print("▶ マクロデータ ダウンロード中（初回のみ）...")
        try:
            frames = {}
            for key, ticker in MACRO_TICKERS.items():
                df = yf.download(ticker, period='10y', progress=False, auto_adjust=True)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).strip().capitalize() for c in df.columns]
                df.index   = pd.to_datetime(df.index, errors='coerce')
                df.index   = df.index.tz_localize(None)
                frames[key] = df['Close'].rename(key)
                time.sleep(0.5)
            if not frames:
                print("[WARN] マクロデータ取得失敗 → 0で埋めます")
                return pd.DataFrame()
            macro_df = pd.concat(frames.values(), axis=1).sort_index().ffill().fillna(0)
            # ※ bfill廃止: look-ahead bias 防止のため初期欠損はゼロ埋め
            joblib.dump(macro_df, MACRO_CACHE_FILE)
            print(f"  -> マクロデータ保存: {len(macro_df)} 日分")
            return macro_df
        except Exception as e:
            print(f"[WARN] マクロデータ取得エラー: {e}")
            return pd.DataFrame()

    def refresh_macro_data(self):
        if os.path.exists(MACRO_CACHE_FILE):
            os.remove(MACRO_CACHE_FILE)
        self.macro_df = self._load_macro_data()

    # ------------------------------------------------------------------
    # モデルロード（シード別 + 旧v5.5フォールバック）
    # ------------------------------------------------------------------
    def _load_or_build_model(self):
        with self.strategy.scope():
            # シード別モデル → 旧v5.5モデルの順で試みる
            search = [
                (self.model_latest, f"最新 seed{self.seed}"),
                (self.model_best,   f"ベスト seed{self.seed}"),
            ]
            # seed=0 の場合は旧v5.5モデルもフォールバック候補に追加
            if self.seed == 0:
                search += [
                    ('./model_v5_5d_latest.keras', "旧v5.5 latest"),
                    ('./model_v5_5d_best.keras',   "旧v5.5 best"),
                ]
            for path, label in search:
                if os.path.exists(path):
                    try:
                        _loss_fn = self._focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
                        model = load_model(
                            path,
                            custom_objects={'focal_loss_fn': _loss_fn}
                        )
                        print(f"▶ {label}モデルをロードしました: {path}")
                        # デュアルヘッドかどうかをログ出力
                        is_dual = self._is_dual_head(model)
                        print(f"  モデル構造: {'デュアルヘッド(prob+ret)' if is_dual else 'シングルヘッド(prob)'}")
                        if os.path.exists(SCALER_FILE):
                            self.scaler     = joblib.load(SCALER_FILE)
                            self.is_trained = True
                            print(f"  スケーラーロード済み: {SCALER_FILE}")
                        return model
                    except Exception as e:
                        print(f"[WARN] {label}モデルのロード失敗 ({path}): {e}")

            print("▶ 既存モデルなし → 新規ビルドします")
            return self._build_model(seed=self.seed)

    # ------------------------------------------------------------------
    # デュアルヘッド判定ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _is_dual_head(model):
        return isinstance(model.output, (list, tuple)) or len(model.outputs) > 1

    # ------------------------------------------------------------------
    # Focal Loss
    # ------------------------------------------------------------------
    @staticmethod
    def _focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):  # [Fix⑧] 定数と統一
        def focal_loss_fn(y_true, y_pred):
            y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            y_true  = tf.cast(y_true, tf.float32)
            alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
            p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
            focal_w = tf.pow(1.0 - p_t, gamma)
            bce     = -tf.math.log(p_t)
            return tf.reduce_mean(alpha_t * focal_w * bce)
        return focal_loss_fn

    # ------------------------------------------------------------------
    # モデル構築（デュアルヘッド）[v6 改善⑤⑥]
    # ------------------------------------------------------------------
    def _build_model(self, seed=0):
        """
        CNN4 + BiLSTM6 + Transformer4 + Dense4 の共有エンコーダ
        + prob_output（sigmoid, Focal Loss）
        + ret_output（linear, Huber Loss）
        seed: 乱数シード（3シードアンサンブル用）
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)

        inp = Input(shape=(self.seq_len, len(self.feature_cols)), name='input')

        # Block A: CNN 4段
        x = Conv1D(128, 11, activation='relu', padding='same', name='conv1')(inp)
        x = BatchNormalization(name='bn_c1')(x)
        x = Conv1D(256,  7, activation='relu', padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn_c2')(x)
        x = MaxPooling1D(2, name='pool1')(x)
        x = Conv1D(512,  5, activation='relu', padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn_c3')(x)
        x = Conv1D(512,  3, activation='relu', padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn_c4')(x)
        x = MaxPooling1D(2, name='pool2')(x)
        x = Dropout(0.2, name='drop_cnn')(x)

        # Block B: BiLSTM 6層
        # [Fix⑤] recurrent_dropout=0.0 に統一 → CuDNN高速カーネルを使用可能にする
        # recurrent_dropout > 0 だとTFはCuDNN非対応の低速実装にフォールバックする
        # 代わりに各BiLSTM層の後にDropoutを追加してregularizationを維持
        lstm_configs = [
            (256, 0.2), (128, 0.2), (64, 0.2),
            (32, 0.15), (16, 0.1), (8, 0.1),
        ]
        for i, (units, drop) in enumerate(lstm_configs, 1):
            x = Bidirectional(
                LSTM(units, return_sequences=True,
                     dropout=0.0, recurrent_dropout=0.0),  # CuDNN互換
                name=f'bilstm{i}'
            )(x)
            x = BatchNormalization(name=f'bn_l{i}')(x)
            x = Dropout(drop, name=f'drop_lstm{i}')(x)    # BN後にDropout

        # Block C: Transformer 4段
        for i in range(1, 5):
            attn = MultiHeadAttention(
                num_heads=8, key_dim=16, dropout=0.1, name=f'attn{i}'
            )(x, x)
            x  = Add(name=f'res{2*i-1}')([x, attn])
            x  = LayerNormalization(name=f'ln{2*i-1}')(x)
            ff = Dense(128, activation='relu', name=f'ff{i}_expand')(x)
            ff = Dropout(0.1, name=f'ff{i}_drop')(ff)
            ff = Dense(16, name=f'ff{i}_proj')(ff)
            x  = Add(name=f'res{2*i}')([x, ff])
            x  = LayerNormalization(name=f'ln{2*i}')(x)

        # Block D: 共有 Dense 4層
        x = GlobalAveragePooling1D(name='gap')(x)
        for i, (units, drop) in enumerate(
            [(256, 0.3), (128, 0.2), (64, 0.15), (32, 0.1)], 1
        ):
            x = Dense(units, activation='relu', name=f'dense{i}')(x)
            if i < 4:
                x = BatchNormalization(name=f'bn_d{i}')(x)
            x = Dropout(drop, name=f'drop_d{i}')(x)

        # [v6 改善⑤] デュアルアウトプットヘッド
        # Head 1: 分類（上昇確率）
        cls_lin    = Dense(1, name='cls_linear')(x)
        prob_output = Activation('sigmoid', dtype='float32', name='prob_output')(cls_lin)

        # Head 2: リターン回帰（±15%クリップ済みリターンを予測）
        ret_lin    = Dense(1, name='ret_linear')(x)
        ret_output = Activation('linear', dtype='float32', name='ret_output')(ret_lin)

        model = Model(
            inputs=inp,
            outputs=[prob_output, ret_output],
            name=f'UltraDeepTrader_v6_5d_s{seed}'
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
            loss={
                'prob_output': self._focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
                'ret_output' : tf.keras.losses.Huber(delta=HUBER_DELTA),
            },
            loss_weights={
                'prob_output': 1.0,
                'ret_output' : LAMBDA_HUBER,
            },
            metrics={
                'prob_output': [
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision',    thresholds=0.5),
                    tf.keras.metrics.Precision(name='precision_40', thresholds=0.4),
                ]
            }
        )
        print(f"\n[v6.0-5d] CNN4+BiLSTM6+Transformer4 | 特徴量{len(self.feature_cols)}個 | デュアルヘッド")
        print(f"  seed={seed}  利益閾値: {PROFIT_THRESHOLD*100:.1f}%  SLIPPAGE: {SLIPPAGE*100:.2f}%")
        model.summary()
        return model

    # ------------------------------------------------------------------
    # 特徴量生成（32個）[v6 改善②③④]
    # ------------------------------------------------------------------
    def _create_features(self, df):
        if len(df) < self.seq_len + 50:
            return pd.DataFrame()
        df = df.copy()
        # ---- インデックスを必ず昇順ソート・重複排除（resample/reindex の前提）----
        df = df[~df.index.duplicated(keep='last')].sort_index()

        # ---- 基本テクニカル（v5.5から継続）----
        df['Ret_1d']     = df['Close'].pct_change(1)
        df['Ret_5d']     = df['Close'].pct_change(5)
        sma20            = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_20_Div'] = (df['Close'] - sma20) / (sma20 + 1e-9)
        df['RSI_14']     = ta.momentum.rsi(df['Close'], window=14) / 100.0
        df['MACD_Hist']  = ta.trend.macd_diff(df['Close']) / (df['Close'] + 1e-9)
        df['Volatility'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
        atr              = ta.volatility.average_true_range(
                               df['High'], df['Low'], df['Close'], window=14)
        df['ATR_Pct']    = atr / (df['Close'] + 1e-9)
        bb_high = ta.volatility.bollinger_hband(df['Close'])
        bb_low  = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Width']   = (bb_high - bb_low) / (sma20 + 1e-9)
        vol_sma20        = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Vol_Zscore'] = (df['Volume'] - vol_sma20) / (
                               df['Volume'].rolling(20).std() + 1e-9)
        df['Ret_20d']    = df['Close'].pct_change(20)
        df['Gap']        = (df['Open'] / (df['Close'].shift(1) + 1e-9)) - 1.0
        obv     = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        obv_sma = obv.rolling(20).mean()
        obv_std = obv.rolling(20).std() + 1e-9
        df['OBV_Norm']   = (obv - obv_sma) / obv_std
        df['Stoch_K']    = ta.momentum.stoch(
                               df['High'], df['Low'], df['Close'], window=14) / 100.0
        df['ADX_14']     = ta.trend.adx(
                               df['High'], df['Low'], df['Close'], window=14) / 100.0
        df['CMF_20']     = ta.volume.chaikin_money_flow(
                               df['High'], df['Low'], df['Close'], df['Volume'], window=20)
        df['Ret_10d']    = df['Close'].pct_change(10)
        sma50            = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_50_Div'] = (df['Close'] - sma50) / (sma50 + 1e-9)
        df['ROC_5']      = ta.momentum.roc(df['Close'], window=5) / 100.0

        # ---- [v6 改善②] 曜日 one-hot + カレンダーアノマリー ----
        # DayOfWeek（線形）を廃止し、以下 5個に置き換え
        # ※ pandas バージョン差を吸収するため np.array() で明示変換
        dow      = np.array(df.index.dayofweek, dtype=np.int32)
        day_vals = np.array(df.index.day,       dtype=np.int32)
        df['Is_Monday']    = (dow == 0).astype(float)
        df['Is_Friday']    = (dow == 4).astype(float)
        # is_month_end / is_month_start: pandas バージョン差を吸収
        try:
            df['Is_Month_End']   = np.array(df.index.is_month_end,   dtype=float)
            df['Is_Month_Start'] = np.array(df.index.is_month_start, dtype=float)
        except Exception:
            # [Fix③] フォールバック: np.roll は配列端を折り返すバグがあるため差分比較方式に変更
            # np.roll(m, -1) → 最終行が先頭行と比較され誤ってIs_Month_End=1になる問題を修正
            _m   = np.array(df.index.month)
            _end = np.zeros(len(_m), dtype=float)
            _end[:-1] = (_m[:-1] != _m[1:]).astype(float)  # 末尾は次月がないため 0
            _sta = np.zeros(len(_m), dtype=float)
            _sta[1:]  = (_m[1:] != _m[:-1]).astype(float)  # 先頭は前月がないため 0
            df['Is_Month_End']   = _end
            df['Is_Month_Start'] = _sta
        # Days_To_SQ: numpy で明示的に clip（pandas Index の .clip() バージョン差を回避）
        df['Days_To_SQ'] = np.clip(day_vals - 14, -5, 5) / 5.0

        # Weekly_Gap（週末ギャップ: 月曜寄り付きと前週金曜終値の乖離）
        fri_close = df['Close'].where(dow == 4).ffill()
        df['Weekly_Gap'] = (df['Open'] - fri_close.shift(1)) / (fri_close.shift(1) + 1e-9)

        # Weekly_ATR_Ratio（週次レンジ vs 日次ATR）
        weekly_range = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
        df['Weekly_ATR_Ratio'] = (weekly_range / (atr + 1e-9)).clip(0, 5) / 5.0

        # ---- [v6 改善③] 週足トレンド（4週SMA比較）----
        # resample → reindex は sort済みインデックスで実行
        # method='ffill' は非単調インデックスでエラーになるため
        # reindex(exact match only) → ffill() の2段構えで安全に処理
        try:
            weekly_close = df['Close'].resample('W-FRI').last()
            weekly_sma4  = weekly_close.rolling(4, min_periods=1).mean()
            weekly_trend = (weekly_close > weekly_sma4).astype(float)
            # reindex は exact match のみ（method指定なし） → その後 ffill
            df['Weekly_Trend'] = (
                weekly_trend.reindex(df.index).ffill().fillna(0.5)
            )
        except Exception:
            # フォールバック: 5日SMAと20日SMAの比較（日足ベース）
            # [Fix⑩] sma20 を上書きしないよう別名 sma20_fb を使用
            sma5_fb  = df['Close'].rolling(5,  min_periods=1).mean()
            sma20_fb = df['Close'].rolling(20, min_periods=1).mean()
            df['Weekly_Trend'] = (sma5_fb > sma20_fb).astype(float)

        # ---- [v6 改善④] 金曜大引け売り圧力 ----
        fri_vol = df['Volume'].where(dow == 4)
        df['Friday_SellPressure'] = (
            (fri_vol / (vol_sma20 + 1e-9))
            .ffill()
            .fillna(1.0)
            .clip(0, 5) / 5.0
        )

        # ---- グローバルマクロ特徴量（v5.5から継続）----
        # ---- グローバルマクロ特徴量: reindex → ffill → fillna(0) ----
        # ※ bfill() は「翌営業日の未来データで過去を埋める」look-ahead biasになるため禁止
        #    初期欠損はゼロ埋め（マクロ情報なし扱い）とする
        if not self.macro_df.empty:
            macro_aligned = self.macro_df.reindex(df.index).ffill().fillna(0)
            if 'nikkei' in macro_aligned.columns:
                df['Nikkei_Ret5d'] = macro_aligned['nikkei'].pct_change(5).fillna(0)
                df['Nikkei_Vol20'] = (macro_aligned['nikkei'].pct_change()
                                      .rolling(20).std().fillna(0))
            else:
                df['Nikkei_Ret5d'] = 0.0
                df['Nikkei_Vol20'] = 0.0
            if 'usdjpy' in macro_aligned.columns:
                df['USDJPY_Ret5d'] = macro_aligned['usdjpy'].pct_change(5).fillna(0)
            else:
                df['USDJPY_Ret5d'] = 0.0
            if 'vix' in macro_aligned.columns:
                df['VIX_Level'] = (macro_aligned['vix'] / 100.0).fillna(0)
            else:
                df['VIX_Level'] = 0.0
        else:
            df['Nikkei_Ret5d'] = 0.0
            df['USDJPY_Ret5d'] = 0.0
            df['VIX_Level']    = 0.0
            df['Nikkei_Vol20'] = 0.0

        # ---- ボラティリティレジーム（v5.5から継続）----
        vol20 = df['Close'].pct_change().rolling(20).std()
        df['Vol_Regime'] = vol20.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)

        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.dropna(subset=self.feature_cols)

    # ------------------------------------------------------------------
    # [v6 改善⑦] ケリー基準ユーティリティ
    # ------------------------------------------------------------------
    def _load_kelly_stats(self):
        if os.path.exists(KELLY_STATS_FILE):
            try:
                with open(KELLY_STATS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        # デフォルト: 勝率50%、損益比1:1（保守スタート）
        return {
            'avg_win_pct' : 2.0,
            'avg_loss_pct': 2.0,
            'win_rate'    : 0.50,
            'n_trades'    : 0,
            'updated_at'  : '—'
        }

    def _save_kelly_stats(self, avg_win, avg_loss, win_rate, n_trades):
        stats = {
            'avg_win_pct' : float(avg_win),
            'avg_loss_pct': float(avg_loss),
            'win_rate'    : float(win_rate),
            'n_trades'    : int(n_trades),
            'updated_at'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(KELLY_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"▶ ケリー統計を保存: {KELLY_STATS_FILE}  "
              f"勝率={win_rate*100:.1f}%  損益比={avg_win/max(avg_loss,0.01):.2f}")

    @staticmethod
    def _kelly_fraction(prob, avg_win_pct, avg_loss_pct,
                        kelly_multiplier=0.5, max_cap=0.30):
        """
        ハーフケリー基準によるポジションサイズ計算。
        prob         : モデル予測確率（0-1）
        avg_win_pct  : 過去の平均利益率（%）
        avg_loss_pct : 過去の平均損失率（%、正の値）
        kelly_multiplier: ケリー分率（保守的に0.5）
        max_cap      : 1銘柄あたりの最大配分率（30%）
        """
        if prob <= 0.5 or avg_loss_pct <= 0:
            return 0.0
        b           = avg_win_pct / max(avg_loss_pct, 0.1)  # 損益比
        q           = 1.0 - prob
        full_kelly  = (b * prob - q) / b
        half_kelly  = full_kelly * kelly_multiplier
        return float(np.clip(half_kelly, 0.0, max_cap))

    # ------------------------------------------------------------------
    # アクション1: データダウンロード
    # ------------------------------------------------------------------
    def download_csv_history(self, years=5):
        print(f"\n=== 【データ一括確保】{years}年分のダウンロード開始 ===")
        print(f"  対象: {len(TRAIN_STOCKS)} 銘柄")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        errors = []
        for i, code in enumerate(TRAIN_STOCKS):
            safe_code = code.replace('/', '_').replace('\\', '_')
            csv_path  = os.path.join(DATA_DIR, f"{safe_code}_hist.csv")
            if i % 100 == 0:
                print(f"  -> 進捗: {i}/{len(TRAIN_STOCKS)} (エラー: {len(errors)})")
            try:
                df = yf.download(code, start=start_date,
                                 end=datetime.now().strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    df.to_csv(csv_path)
                else:
                    errors.append(f"{code} (No data)")
            except Exception as e:
                errors.append(f"{code} ({e})")
            time.sleep(0.3)
        print(f"\n▶ ダウンロード完了: 成功 {len(TRAIN_STOCKS)-len(errors)} / エラー {len(errors)}")
        if errors:
            print(f"  エラーサンプル: {errors[:10]}")

    # ------------------------------------------------------------------
    # アクション2: 前処理 → チャンク npz 保存
    # ------------------------------------------------------------------
    def preprocess_to_npz(self):
        csv_files = glob.glob(os.path.join(DATA_DIR, "*_hist.csv"))
        if not csv_files:
            return print("▶ エラー: CSVが見つかりません。先に download を実行してください。")
        random.shuffle(csv_files)
        print(f"\n=== 【前処理】{len(csv_files)} ファイル → チャンク npz ===")
        print(f"  スリッページ設定: {SLIPPAGE*100:.2f}% (v6 改善①)")

        # --- フェーズ1: スケーラー + class_weight 構築 ---
        need_cw = (self.class_weight_ratio is None) and (not os.path.exists(CLASS_WEIGHT_FILE))
        if os.path.exists(SCALER_FILE) and not need_cw:
            print("  既存スケーラー・class_weight を再利用します。")
            self.scaler     = joblib.load(SCALER_FILE)
            self.is_trained = True
            if self.class_weight_ratio is None and os.path.exists(CLASS_WEIGHT_FILE):
                with open(CLASS_WEIGHT_FILE, 'r') as cf:
                    self.class_weight_ratio = {int(k): float(v)
                                               for k, v in json.load(cf).items()}
        else:
            print("  ステップ1: スケーラー・class_weight 構築中（150社サンプリング）...")
            sample_dfs = [] if not os.path.exists(SCALER_FILE) else None
            all_labels = []
            _err_count  = 0   # 最初の5件だけ詳細表示
            _ok_count   = 0
            for f in random.sample(csv_files, min(150, len(csv_files))):
                try:
                    df = read_stock_csv(f)
                    if df.empty: continue
                    fd = self._create_features(df)
                    if not fd.empty:
                        _ok_count += 1
                        if sample_dfs is not None:
                            sample_dfs.append(fd[self.feature_cols])
                        no   = fd['Open'].shift(-1).values
                        nc   = fd['Close'].shift(-HORIZON_SHIFT).values
                        lbl, _, _ = compute_labels_and_returns(no, nc)
                        all_labels.extend(lbl[~np.isnan(lbl)].tolist())
                except Exception as e:
                    _err_count += 1
                    if _err_count <= 5:
                        import traceback as _tb
                        print(f"  [WARN-SAMPLE] {os.path.basename(f)}: "
                              f"{type(e).__name__}: {e}")
                        _tb.print_exc()
            print(f"  -> サンプリング結果: 成功 {_ok_count} / エラー {_err_count} / 計150")
            if sample_dfs:
                self.scaler.fit(pd.concat(sample_dfs))
                joblib.dump(self.scaler, SCALER_FILE)
                self.is_trained = True
                print(f"  -> スケーラー保存: {SCALER_FILE}")
            elif not os.path.exists(SCALER_FILE):
                print("[ERROR] スケーラー構築用データが不足しています。")
                print("  → 上記の [WARN-SAMPLE] エラーを確認してください。")
                return  # スケーラー未構築ならフェーズ2も意味なし
            else:
                self.scaler     = joblib.load(SCALER_FILE)
                self.is_trained = True
            if all_labels:
                all_labels_arr = np.array(all_labels)
                if len(np.unique(all_labels_arr)) == 2:
                    weights = compute_class_weight('balanced',
                                                   classes=np.array([0, 1]),
                                                   y=all_labels_arr)
                    self.class_weight_ratio = {0: float(weights[0]), 1: float(weights[1])}
                else:
                    self.class_weight_ratio = {0: 1.0, 1: 1.0}
                with open(CLASS_WEIGHT_FILE, 'w') as cf:
                    json.dump({str(k): v for k, v in self.class_weight_ratio.items()}, cf, indent=2)
                print(f"  -> class_weight: 0={self.class_weight_ratio[0]:.3f} / "
                      f"1={self.class_weight_ratio[1]:.3f}")
                pos_rate = sum(all_labels) / len(all_labels)
                print(f"  -> 正例率（スリッページ後）: {pos_rate*100:.1f}%  "
                      f"（旧モデルより低くなるのが正常）")

        # --- フェーズ2: サンプル生成 → Train/Val 時系列分割チャンク保存 ---
        # [Fix9] TRAIN_CUTOFF_DATE を境に train/ と val/ サブディレクトリへ物理分割
        # → train_from_npz が本物のウォークフォワード検証を行える
        if not self.is_trained:
            return print("[ERROR] スケーラーが未構築です。フェーズ1のエラーを確認してください。")

        print("  ステップ2: 特徴量シーケンスを生成・Train/Val時系列分割チャンク保存中...")
        print(f"  Train: ～{TRAIN_CUTOFF_DATE}  Val: {VAL_START_DATE}～")
        SAMPLES_PER_CHUNK = 50_000
        train_cid = val_cid = 0
        buf_tr_X, buf_tr_y, buf_tr_ret, buf_tr_w = [], [], [], []
        buf_va_X, buf_va_y, buf_va_ret, buf_va_w = [], [], [], []
        total_tr = total_va = processed = skipped = 0
        _p2_err_count = 0
        cutoff_ts = pd.Timestamp(TRAIN_CUTOFF_DATE)

        def flush_chunk_dir(X_list, y_list, ret_list, w_list, cid, dirpath, tag):
            X_arr   = np.array(X_list,   dtype=np.float32)
            y_arr   = np.array(y_list,   dtype=np.float32)
            ret_arr = np.array(ret_list, dtype=np.float32)
            w_arr   = np.array(w_list,   dtype=np.float32)
            idx     = np.random.permutation(len(X_arr))
            X_arr, y_arr, ret_arr, w_arr = X_arr[idx], y_arr[idx], ret_arr[idx], w_arr[idx]
            path = os.path.join(dirpath, f"{CHUNK_PREFIX}{cid:04d}.npz")
            np.savez_compressed(path, X=X_arr, y=y_arr, y_ret=ret_arr, y_w=w_arr)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  -> [{tag}] chunk_{cid:04d}.npz: {len(X_arr):,}サンプル ({size_mb:.1f}MB)")
            return cid + 1, [], [], [], []

        for i, file in enumerate(csv_files):
            if train_cid >= MAX_CHUNKS and val_cid >= MAX_CHUNKS:
                print(f"  ⚠️  最大チャンク数 {MAX_CHUNKS} に達しました。")
                break
            if i % 200 == 0:
                print(f"  -> 進捗: {i}/{len(csv_files)} "
                      f" Train{total_tr:,} Val{total_va:,} サンプル")
            try:
                df = read_stock_csv(file)
                if df.empty:
                    skipped += 1; continue
                f_df = self._create_features(df)
                if f_df.empty or len(f_df) < self.seq_len + HORIZON_SHIFT + 1:
                    skipped += 1; continue

                feat_vals  = self.scaler.transform(f_df[self.feature_cols].values)
                next_open  = f_df['Open'].shift(-1).values
                next_close = f_df['Close'].shift(-HORIZON_SHIFT).values
                tgt_v, w_v, ret_v = compute_labels_and_returns(next_open, next_close)
                dates      = f_df.index  # DatetimeIndex（日付情報）

                max_start = len(feat_vals) - self.seq_len - HORIZON_SHIFT - 1
                for start in range(0, max_start, 3):
                    end     = start + self.seq_len
                    tgt     = tgt_v[end - 1]
                    end_date = dates[end - 1]  # このシーケンスの最終日
                    if not np.isnan(tgt):
                        sample = (feat_vals[start:end], float(tgt),
                                  float(ret_v[end-1]), float(w_v[end-1]))
                        if end_date <= cutoff_ts:
                            # Train データ（過去）
                            buf_tr_X.append(sample[0]); buf_tr_y.append(sample[1])
                            buf_tr_ret.append(sample[2]); buf_tr_w.append(sample[3])
                            total_tr += 1
                        else:
                            # Val データ（未来）
                            buf_va_X.append(sample[0]); buf_va_y.append(sample[1])
                            buf_va_ret.append(sample[2]); buf_va_w.append(sample[3])
                            total_va += 1
                processed += 1
            except Exception as e:
                skipped += 1
                if _p2_err_count < 3:
                    print(f"  [WARN-P2] {os.path.basename(file)}: {type(e).__name__}: {e}")
                    _p2_err_count += 1

            if len(buf_tr_X) >= SAMPLES_PER_CHUNK:
                train_cid, buf_tr_X, buf_tr_y, buf_tr_ret, buf_tr_w = flush_chunk_dir(
                    buf_tr_X, buf_tr_y, buf_tr_ret, buf_tr_w,
                    train_cid, CHUNK_TRAIN_DIR, "TRAIN")
            if len(buf_va_X) >= SAMPLES_PER_CHUNK:
                val_cid, buf_va_X, buf_va_y, buf_va_ret, buf_va_w = flush_chunk_dir(
                    buf_va_X, buf_va_y, buf_va_ret, buf_va_w,
                    val_cid, CHUNK_VAL_DIR, "VAL  ")

        if buf_tr_X:
            train_cid, buf_tr_X, buf_tr_y, buf_tr_ret, buf_tr_w = flush_chunk_dir(
                buf_tr_X, buf_tr_y, buf_tr_ret, buf_tr_w,
                train_cid, CHUNK_TRAIN_DIR, "TRAIN")
        if buf_va_X:
            val_cid, buf_va_X, buf_va_y, buf_va_ret, buf_va_w = flush_chunk_dir(
                buf_va_X, buf_va_y, buf_va_ret, buf_va_w,
                val_cid, CHUNK_VAL_DIR, "VAL  ")

        meta = {
            "seq_len"            : self.seq_len,
            "n_features"         : len(self.feature_cols),
            "feature_cols"       : self.feature_cols,
            "total_train_samples": total_tr,
            "total_val_samples"  : total_va,
            "train_chunks"       : train_cid,
            "val_chunks"         : val_cid,
            "train_cutoff"       : TRAIN_CUTOFF_DATE,
            "processed"          : processed,
            "skipped"            : skipped,
            "profit_threshold"   : PROFIT_THRESHOLD,
            "slippage"           : SLIPPAGE,
            "class_weight_ratio" : self.class_weight_ratio,
            "created_at"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for d in [CHUNK_DIR, CHUNK_TRAIN_DIR, CHUNK_VAL_DIR]:
            with open(os.path.join(d, "meta.json"), 'w') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        pos_rate = total_va / (total_tr + total_va + 1e-9)
        print(f"\n{'='*60}")
        print(f"✅ 前処理完了！（時系列分割版）")
        print(f"  Trainサンプル : {total_tr:,}  ({train_cid}チャンク) ～{TRAIN_CUTOFF_DATE}")
        print(f"  Val  サンプル : {total_va:,}  ({val_cid}チャンク)  {VAL_START_DATE}～")
        print(f"  Val比率       : {pos_rate*100:.1f}%  (目安: 20-30%)")
        print(f"  処理銘柄      : {processed} / スキップ: {skipped}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # アクション3: train_kaggle（チャンクストリーミング学習）
    # ------------------------------------------------------------------
    def train_from_npz(self, npz_path=None):
        chunk_dir = self._find_chunk_dir(npz_path)
        if chunk_dir is None:
            return

        # [Fix9] Train/Val サブディレクトリを優先して使用
        train_subdir = os.path.join(chunk_dir, "train")
        val_subdir   = os.path.join(chunk_dir, "val")
        use_split    = (os.path.isdir(train_subdir)
                        and glob.glob(os.path.join(train_subdir, f"{CHUNK_PREFIX}*.npz")))
        if use_split:
            chunk_files      = sorted(glob.glob(os.path.join(train_subdir, f"{CHUNK_PREFIX}*.npz")))
            val_chunk_files  = sorted(glob.glob(os.path.join(val_subdir,   f"{CHUNK_PREFIX}*.npz")))
            print(f"  [時系列分割モード] Train:{len(chunk_files)}チャンク / Val:{len(val_chunk_files)}チャンク")
        else:
            chunk_files      = sorted(glob.glob(os.path.join(chunk_dir, f"{CHUNK_PREFIX}*.npz")))
            val_chunk_files  = None
            print(f"  [従来モード] ランダムchunk分割 (時系列分割はpreprocess再実行で有効化)")
        if not chunk_files:
            return print(f"▶ エラー: チャンクが見つかりません ({chunk_dir})")

        n_gpu = len(tf.config.list_physical_devices('GPU'))
        print(f"\n=== 【Kaggle学習 / シード{self.seed}】===")
        print(f"  チャンク数: {len(chunk_files)}  GPU: {n_gpu}台")

        # スケーラーロード
        scaler_search = (
            [SCALER_FILE,
             os.path.join(chunk_dir, os.path.basename(SCALER_FILE))]
            + glob.glob(f'/kaggle/input/**/{os.path.basename(SCALER_FILE)}', recursive=True)
        )
        for sc in scaler_search:
            if os.path.exists(sc):
                self.scaler     = joblib.load(sc)
                self.is_trained = True
                print(f"  スケーラーロード: {sc}")
                break
        else:
            # [Fix①] ダブルスケーリングバグ防止
            # チャンクのX はpreprocess時にscaler.transform済み → 再fitすると二重スケーリングになる
            # scaler.pkl が見つからない = preprocessが正常完了していない → 学習中止
            print("[ERROR] scaler.pkl が見つかりません。")
            print("  チャンクのXはスケーリング済みのため、チャンクからscalerを推定するとダブルスケーリングになります。")
            print("  先に preprocess を実行して scaler_v6_5d.pkl を生成してください。")
            return

        # class_weight ロード
        if self.class_weight_ratio is None:
            cw_search = (
                [CLASS_WEIGHT_FILE]
                + glob.glob(f'/kaggle/input/**/{os.path.basename(CLASS_WEIGHT_FILE)}', recursive=True)
                + glob.glob(os.path.join(chunk_dir, '*.json'))
            )
            for cw_path in cw_search:
                if not os.path.exists(cw_path):
                    continue
                try:
                    with open(cw_path, 'r') as f:
                        data = json.load(f)
                    cw_data = data.get('class_weight_ratio', data)
                    if cw_data and ('0' in cw_data or 0 in cw_data):
                        self.class_weight_ratio = {int(k): float(v) for k, v in cw_data.items()}
                        print(f"  class_weight: {self.class_weight_ratio}")
                        break
                except Exception:
                    pass
            if self.class_weight_ratio is None:
                # meta.json フォールバック
                for mp in glob.glob(os.path.join(chunk_dir, '**/meta.json'), recursive=True):
                    try:
                        with open(mp, 'r') as f:
                            meta = json.load(f)
                        cw_data = meta.get('class_weight_ratio')
                        if cw_data:
                            self.class_weight_ratio = {int(k): float(v) for k, v in cw_data.items()}
                            break
                    except Exception:
                        pass
            if self.class_weight_ratio is None:
                self.class_weight_ratio = {0: 1.0, 1: 1.0}
                print("[WARN] class_weight ファイルが見つかりません。均等ウェイトを使用します。")
        cw = self.class_weight_ratio

        # チャンク shape 確認
        probe      = np.load(chunk_files[0])
        chunk_size = probe['X'].shape[0]
        del probe

        # Train / Val 分割
        if use_split:
            # [Fix9] 時系列分割済みの場合: train/ と val/ をそのまま使用
            train_files = chunk_files
            val_files   = val_chunk_files if val_chunk_files else chunk_files[-1:]
        else:
            # 従来: ランダムchunk分割（先頭20%をval）
            val_chunk_count = max(1, int(len(chunk_files) * 0.2))
            val_files       = chunk_files[:val_chunk_count]
            train_files     = chunk_files[val_chunk_count:]
            if not train_files:
                train_files = chunk_files
        print(f"  Trainチャンク: {len(train_files)}  Valチャンク: {len(val_files)}")

        # 再開エポック決定
        progress         = load_progress(self.progress_file)
        completed        = progress["completed_epochs"]
        initial_epoch    = completed
        remaining_epochs = self.epochs - completed
        if remaining_epochs <= 0:
            print(f"▶ すでに {completed} エポック完了。--epochs を増やすか --reset で再実行してください。")
            return
        print(f"▶ シード{self.seed}: エポック {completed+1} から開始  残り: {remaining_epochs}")

        n_features = len(self.feature_cols)
        seq_len    = self.seq_len

        # [Fix3] chunk-level generator + flat_map(from_tensor_slices)
        # 旧: per-sample yield → Python GILボトルネックで低速
        # 新: チャンク単位でまとめてnumpy配列をyield → flat_mapでTF内部処理
        #     → CPUデータ準備とGPU演算がパイプライン化され大幅に高速化
        cw0_f = float(cw.get(0, 1.0))
        cw1_f = float(cw.get(1, 1.0))

        def _chunk_array_generator(file_list, shuffle_files, shuffle_samples):
            """チャンクファイルをnumpy配列ごとyield（per-sampleではない）"""
            files = list(file_list)
            if shuffle_files:
                random.shuffle(files)
            for path in files:
                d     = np.load(path)
                X_c   = d['X'].astype(np.float32)
                y_c   = d['y'].astype(np.float32)
                ret_c = np.clip(
                    d['y_ret'].astype(np.float32) if 'y_ret' in d
                    else np.zeros(len(y_c), dtype=np.float32),
                    -0.15, 0.15
                )
                w_c   = (
                    d['y_w'].astype(np.float32) if 'y_w' in d
                    else np.clip(np.abs(ret_c) * 10, 0.3, 5.0).astype(np.float32)
                )
                del d
                class_w = np.where(y_c == 1, cw1_f, cw0_f).astype(np.float32)
                w_c     = (w_c * class_w).astype(np.float32)
                del class_w
                if shuffle_samples:
                    idx = np.random.permutation(len(X_c))
                    X_c, y_c, ret_c, w_c = X_c[idx], y_c[idx], ret_c[idx], w_c[idx]
                yield X_c, y_c, ret_c, w_c
                del X_c, y_c, ret_c, w_c

        def build_dataset(file_list, shuffle_files=True, shuffle_samples=True):
            # チャンク配列のデータセット（各要素が1チャンク分のnumpy配列）
            chunk_ds = tf.data.Dataset.from_generator(
                lambda: _chunk_array_generator(file_list, shuffle_files, shuffle_samples),
                output_signature=(
                    tf.TensorSpec(shape=(None, seq_len, n_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,),                     dtype=tf.float32),
                    tf.TensorSpec(shape=(None,),                     dtype=tf.float32),
                    tf.TensorSpec(shape=(None,),                     dtype=tf.float32),
                )
            )
            # flat_map: チャンク配列 → per-sampleデータセットに展開
            # from_tensor_slices はTF内部のC++実装のため GILの影響を受けない
            def chunk_to_samples(X_c, y_c, ret_c, w_c):
                return tf.data.Dataset.from_tensor_slices((
                    X_c,
                    {'prob_output': y_c, 'ret_output': ret_c},
                    w_c
                ))
            return chunk_ds.flat_map(chunk_to_samples)

        bsz                  = self.effective_batch
        approx_val_samples   = len(val_files) * chunk_size
        approx_train_samples = len(train_files) * chunk_size
        val_steps            = max(1, approx_val_samples   // bsz)
        steps_per_epoch      = max(1, approx_train_samples // bsz)
        print(f"  steps_per_epoch: ~{steps_per_epoch}  val_steps: ~{val_steps}")

        print("  Valデータセット構築中...")
        val_ds = build_dataset(val_files, shuffle_files=False, shuffle_samples=False)
        val_ds = val_ds.batch(bsz).prefetch(tf.data.AUTOTUNE)

        # ---------------------------------------------------------------
        # 学習率の初期値をバッチサイズに比例スケール（Linear Scaling Rule）
        # 基準: batch=256 / 1GPU → LR=0.0003
        # 例:   batch=256 / 2GPU → effective=512 → LR=0.0006
        # ---------------------------------------------------------------
        BASE_LR    = 0.0003
        BASE_BATCH = 256
        current_lr = BASE_LR * (bsz / BASE_BATCH)
        current_lr = float(np.clip(current_lr, BASE_LR, BASE_LR * 4))  # 最大4倍まで
        with self.strategy.scope():
            self.model.optimizer.learning_rate.assign(current_lr)

        # Early Stopping / LR Reduction（手動管理）
        # patience を長めに設定：val_aucは揺れやすいため短いと早期収束する
        es_patience       = 15          # 旧: 8  → 15に延長
        es_wait           = 0
        es_best_val_auc   = progress['best_val_auc']
        lr_patience       = 6           # 旧: 4  → 6に延長
        lr_wait           = 0
        lr_min            = 1e-7

        # モード崩壊検知用カウンタ
        # val_auc が COLLAPSE_THRESHOLD 以下のエポックが COLLAPSE_PATIENCE 連続したら
        # LRをリセットしてベストモデルに巻き戻す
        COLLAPSE_THRESHOLD = 0.505      # AUC≦0.505 = ほぼランダム予測
        COLLAPSE_PATIENCE  = 3
        collapse_wait      = 0
        collapse_lr_reset  = current_lr * 1.5  # 崩壊後のリセット先LR

        print(f"\n▶ v6.0-5d シード{self.seed} 学習開始...")
        print(f"  初期LR: {current_lr:.4e}  (バッチ{bsz}に比例スケール済み)")
        print(f"  EarlyStopping patience={es_patience}  "
              f"ReduceLR patience={lr_patience}")
        print(f"  モード崩壊検知: val_auc≦{COLLAPSE_THRESHOLD} が"
              f" {COLLAPSE_PATIENCE}エポック連続でリセット")

        for epoch_idx in range(initial_epoch, self.epochs):
            print(f"\n--- エポック {epoch_idx + 1}/{self.epochs} ---")
            train_ds = build_dataset(train_files, shuffle_files=True, shuffle_samples=True)
            train_ds = train_ds.batch(bsz).prefetch(tf.data.AUTOTUNE)

            callbacks = [
                TrainingProgressCallback(
                    initial_epoch=epoch_idx,
                    total_epochs=self.epochs,
                    steps_per_epoch=steps_per_epoch,
                    model_best_path=self.model_best,
                    model_latest_path=self.model_latest,
                    progress_file=self.progress_file,
                ),
            ]

            hist = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epoch_idx + 1,
                initial_epoch=epoch_idx,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=0,
            )

            # デュアルヘッドのメトリクスキー
            val_auc = float((
                hist.history.get('val_prob_output_auc') or
                hist.history.get('val_auc') or [0.0]
            )[-1])

            # ── モード崩壊検知 ──────────────────────────────────────────
            if val_auc <= COLLAPSE_THRESHOLD:
                collapse_wait += 1
                if collapse_wait >= COLLAPSE_PATIENCE:
                    # ベストモデルが存在すれば巻き戻す
                    if os.path.exists(self.model_best):
                        print(f"  ⚠️  モード崩壊検知 ({collapse_wait}エポック連続 "
                              f"val_auc≦{COLLAPSE_THRESHOLD})。"
                              f"  ベストモデルに巻き戻してLRリセット → {collapse_lr_reset:.2e}")
                        try:
                            _loss_fn = self._focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
                            with self.strategy.scope():
                                self.model = load_model(
                                    self.model_best,
                                    custom_objects={'focal_loss_fn': _loss_fn}
                                )
                                self.model.optimizer.learning_rate.assign(collapse_lr_reset)
                        except Exception as e:
                            print(f"  [WARN] 巻き戻し失敗: {e}")
                        collapse_wait     = 0
                        lr_wait           = 0
                        collapse_lr_reset = max(collapse_lr_reset * 0.5, lr_min * 10)
                    else:
                        print(f"  ⚠️  モード崩壊検知。ベストモデルが未存在のため"
                              f" LRのみリセット → {collapse_lr_reset:.2e}")
                        with self.strategy.scope():
                            self.model.optimizer.learning_rate.assign(collapse_lr_reset)
                        collapse_wait = 0
            else:
                collapse_wait = 0

            # ── EarlyStopping & ReduceLR ────────────────────────────────
            if val_auc > es_best_val_auc:
                es_best_val_auc   = val_auc
                es_wait           = 0
                lr_wait           = 0
            else:
                es_wait += 1
                lr_wait += 1

            if lr_wait >= lr_patience:
                current_lr = max(current_lr * 0.5, lr_min)
                print(f"  ReduceLR → {current_lr:.2e}")
                with self.strategy.scope():
                    self.model.optimizer.learning_rate.assign(current_lr)
                lr_wait = 0

            if es_wait >= es_patience:
                completed_ep = load_progress(self.progress_file)['completed_epochs']
                print(f"▶ EarlyStopping 発動 → エポック{completed_ep}で終了  "
                      f"BESTモデル: {self.model_best}")
                break

            # [Fix1] clear_session() + model reload を廃止
            # 旧: 毎エポック clear_session → load_model でAdamのmomentumが完全リセットされ学習不安定
            # 新: train_dsのみ解放。Adamの内部状態（m/v）はモデル内に保持し続ける
            del train_ds
            gc.collect()

            # Val データセットも再構築（メモリ効率のため）
            val_ds = build_dataset(val_files, shuffle_files=False, shuffle_samples=False)
            val_ds = val_ds.batch(bsz).prefetch(tf.data.AUTOTUNE)

        final_prog = load_progress(self.progress_file)
        print(f"\n✅ シード{self.seed} 訓練完了")
        print(f"  完了エポック: {final_prog['completed_epochs']}")
        print(f"  最高AUC     : {final_prog['best_val_auc']:.4f}")
        print(f"  BESTモデル  : {self.model_best}")
        if self.seed < N_ENSEMBLE - 1:
            print(f"\n  次のステップ: --seed {self.seed + 1} で次のモデルを訓練してください")
        else:
            print(f"\n  全{N_ENSEMBLE}シード訓練完了！morning でアンサンブル予測を実行できます。")

    # ------------------------------------------------------------------
    # アクション4: train（CSV直接学習）
    # ------------------------------------------------------------------
    def train_from_csv(self):
        csv_files = glob.glob(os.path.join(DATA_DIR, "*_hist.csv"))
        if not csv_files:
            return print("▶ エラー: CSVが見つかりません。")
        random.shuffle(csv_files)
        val_split   = int(len(csv_files) * 0.8)
        train_files = csv_files[:val_split]
        val_files   = csv_files[val_split:]
        print(f"▶ データ分割: Train {len(train_files)} / Val {len(val_files)}")

        if not self.is_trained:
            print("▶ スケーラー・class_weight 構築中...")
            sample_dfs = []
            all_labels = []
            for f in random.sample(train_files, min(100, len(train_files))):
                try:
                    df = read_stock_csv(f)
                    if df.empty: continue
                    fd = self._create_features(df)
                    if not fd.empty:
                        sample_dfs.append(fd[self.feature_cols])
                        no  = fd['Open'].shift(-1).values
                        nc  = fd['Close'].shift(-HORIZON_SHIFT).values
                        lbl, _, _ = compute_labels_and_returns(no, nc)
                        all_labels.extend(lbl[~np.isnan(lbl)].tolist())
                except Exception:
                    pass
            if not sample_dfs:
                return print("▶ エラー: スケーラー構築用データが不足しています。")
            self.scaler.fit(pd.concat(sample_dfs))
            joblib.dump(self.scaler, SCALER_FILE)
            self.is_trained = True
            if all_labels:
                all_labels_arr = np.array(all_labels)
                if len(np.unique(all_labels_arr)) == 2:
                    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_labels_arr)
                    self.class_weight_ratio = {0: float(weights[0]), 1: float(weights[1])}
                else:
                    self.class_weight_ratio = {0: 1.0, 1: 1.0}
                with open(CLASS_WEIGHT_FILE, 'w') as cf:
                    json.dump({str(k): v for k, v in self.class_weight_ratio.items()}, cf)

        progress         = load_progress(self.progress_file)
        completed        = progress["completed_epochs"]
        initial_epoch    = completed
        remaining_epochs = self.epochs - completed
        if remaining_epochs <= 0:
            print(f"▶ すでに {completed} エポック完了。--epochs を増やして再実行してください。")
            return
        if completed > 0:
            print(f"▶ エポック {completed+1} から再開します")

        print(f"\n▶ v6.0-5d シード{self.seed} CSVモード訓練")
        print(f"   seq_len={self.seq_len}  残り: {remaining_epochs} エポック")

        train_gen = StockDataGenerator(
            train_files, self, self.seq_len,
            batch_size=self.batch_size, is_training=True,
            date_cutoff=TRAIN_CUTOFF_DATE
        )
        val_gen = StockDataGenerator(
            val_files, self, self.seq_len,
            batch_size=self.batch_size, is_training=False,
            date_cutoff=TRAIN_CUTOFF_DATE
        )
        csv_steps = len(train_gen)
        print(f"  Walk-Forward: Train≤{TRAIN_CUTOFF_DATE} / Val≥{VAL_START_DATE}")
        print(f"  Trainサンプル: {len(train_gen.samples):,}  Valサンプル: {len(val_gen.samples):,}")

        # デュアルヘッドのモニターキー
        monitor_metric = 'val_prob_output_auc'

        callbacks = [
            TrainingProgressCallback(
                initial_epoch=initial_epoch,
                total_epochs=self.epochs,
                steps_per_epoch=csv_steps,
                model_best_path=self.model_best,
                model_latest_path=self.model_latest,
                progress_file=self.progress_file,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric, mode='max',
                patience=5, restore_best_weights=True,
                start_from_epoch=initial_epoch
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, mode='max',
                factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
        ]

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=0,
        )
        print(f"\n✅ シード{self.seed} CSVモード訓練完了。")

    # ------------------------------------------------------------------
    # チャンクディレクトリ探索ユーティリティ
    # ------------------------------------------------------------------
    def _find_chunk_dir(self, hint=None):
        if hint and os.path.isdir(hint):
            return hint
        if hint and os.path.isfile(hint):
            return os.path.dirname(hint)
        for meta in glob.glob('/kaggle/input/**/meta.json', recursive=True):
            d = os.path.dirname(meta)
            if glob.glob(os.path.join(d, f"{CHUNK_PREFIX}*.npz")):
                print(f"[INFO] Kaggle Dataset 自動検出: {d}")
                return d
        if glob.glob(os.path.join(CHUNK_DIR, f"{CHUNK_PREFIX}*.npz")):
            return CHUNK_DIR
        print("▶ エラー: チャンクが見つかりません。先に preprocess を実行してください。")
        return None

    # ------------------------------------------------------------------
    # アクション5: morning（朝の予測スキャン）[v6 改善⑥⑦]
    # ------------------------------------------------------------------
    def morning_predict(self, top_n=5):
        if not self.is_trained:
            return print("▶ モデルが初期化されていません！先に train を実行してください。")

        # [v6 改善⑥] 全シードモデルをアンサンブルロード
        _loss_fn = self._focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
        ensemble_models = []
        for i in range(N_ENSEMBLE):
            for path in [MODEL_SEEDS_BEST[i], MODEL_SEEDS_LATEST[i]]:
                if os.path.exists(path):
                    try:
                        m = load_model(path, custom_objects={'focal_loss_fn': _loss_fn})
                        ensemble_models.append((i, m))
                        print(f"▶ アンサンブル: seed{i} モデルをロード ({path})")
                        break
                    except Exception as e:
                        print(f"[WARN] seed{i} モデルロード失敗: {e}")

        # フォールバック: 旧v5.5モデルまたはメモリ内モデル
        if not ensemble_models:
            for path in ('./model_v5_5d_best.keras', './model_v5_5d_latest.keras'):
                if os.path.exists(path):
                    try:
                        m = load_model(path, custom_objects={'focal_loss_fn': _loss_fn})
                        ensemble_models.append((0, m))
                        print(f"▶ フォールバック: 旧v5.5モデルを使用 ({path})")
                        break
                    except Exception:
                        pass
        if not ensemble_models:
            ensemble_models.append((0, self.model))
            print(f"▶ メモリ内モデルを使用 (seed{self.seed})")

        print(f"\n▶ アンサンブル構成: {len(ensemble_models)}モデル "
              f"(seeds: {[s for s,_ in ensemble_models]})")

        # [v6 改善⑦] ケリー統計ロード
        kelly_stats = self._load_kelly_stats()
        kelly_ok    = kelly_stats['n_trades'] >= 10  # 10取引以上で信頼性あり
        print(f"▶ ケリー統計: 累計{kelly_stats['n_trades']}取引  "
              f"勝率={kelly_stats['win_rate']*100:.1f}%  "
              f"損益比={kelly_stats['avg_win_pct']/max(kelly_stats['avg_loss_pct'],0.01):.2f}"
              f"  {'✅有効' if kelly_ok else '⚠️ 10取引未満（デフォルト値）'}")

        lookback_days = max(400, int((self.seq_len + 80) * 365 / 250) + 60)
        start_date    = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        print(f"\n===[朝の予測スキャン] v6.0-5d ===")
        print(f"  予測対象: {len(PREDICT_STOCKS)} 銘柄 / 過去{lookback_days}日")
        print(f"  スリッページ: {SLIPPAGE*100:.2f}%  利益閾値: {PROFIT_THRESHOLD*100:.1f}%超")

        predictions_pool = []
        skipped = 0

        for i, code in enumerate(PREDICT_STOCKS):
            if i % 50 == 0 and i > 0:
                print(f"  -> 進捗: {i}/{len(PREDICT_STOCKS)} (スキップ: {skipped})")
            try:
                raw_df = yf.download(code, start=start_date, progress=False, auto_adjust=True)
                if raw_df.empty: skipped += 1; continue
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                raw_df.columns = [str(c).strip().capitalize() for c in raw_df.columns]
                raw_df = raw_df.drop(columns=['Price'], errors='ignore')
                raw_df.index = pd.to_datetime(raw_df.index, errors='coerce')
                raw_df = raw_df[raw_df.index.notna()].copy()
                raw_df.index = raw_df.index.tz_localize(None)

                df_feat = self._create_features(raw_df)
                if df_feat.empty or len(df_feat) < self.seq_len:
                    skipped += 1; continue

                scaled_feats  = self.scaler.transform(df_feat[self.feature_cols].values)
                last_seq      = np.expand_dims(scaled_feats[-self.seq_len:], axis=0)
                ref_price     = float(df_feat['Close'].iloc[-1])
                ref_date_str  = str(df_feat.index[-1].date())

                # [v6 改善⑥] アンサンブル予測
                all_probs = []
                all_rets  = []
                for _, m in ensemble_models:
                    pred = m.predict(last_seq, verbose=0)
                    if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                        # デュアルヘッド: np.squeeze で shape 差異を吸収 [Fix7]
                        all_probs.append(float(np.squeeze(pred[0])))
                        all_rets.append(float(np.squeeze(pred[1])))
                    else:
                        # シングルヘッド（旧モデル）
                        all_probs.append(float(np.squeeze(pred)))
                        all_rets.append(0.0)

                prob_up    = float(np.mean(all_probs))
                exp_ret    = float(np.mean(all_rets))   # 回帰ヘッドの平均予測リターン
                prob_std   = float(np.std(all_probs)) if len(all_probs) > 1 else 0.0

                # [v6 改善⑦] ケリー配分計算
                kelly_frac = self._kelly_fraction(
                    prob_up,
                    kelly_stats['avg_win_pct'],
                    kelly_stats['avg_loss_pct']
                )

                predictions_pool.append({
                    "code"       : code,
                    "prob"       : prob_up,
                    "exp_ret"    : exp_ret,
                    "prob_std"   : prob_std,
                    "kelly_pct"  : kelly_frac,
                    "ref_price"  : ref_price,
                    "ref_date"   : ref_date_str,
                    "seq_memory" : scaled_feats[-self.seq_len:]
                })
            except Exception as e:
                skipped += 1
                if skipped <= 5:  # [Fix⑦] 最初の5件だけエラー詳細を表示
                    print(f"  [WARN] {code}: {type(e).__name__}: {e}")

        if not predictions_pool:
            return print("⚠️ 予測対象が0件です。")

        all_probs_arr = np.array([x['prob'] for x in predictions_pool])
        print(f"\n[ スコア分布 ({len(all_probs_arr)}銘柄) ]")
        for label, pct in [("10%ile", 10), ("25%ile", 25), ("中央値", 50),
                            ("75%ile", 75), ("90%ile", 90)]:
            print(f"  {label}: {float(np.percentile(all_probs_arr, pct))*100:.2f}%")
        print(f"  平均: {all_probs_arr.mean()*100:.2f}%  最大: {all_probs_arr.max()*100:.2f}%")

        sorted_all     = sorted(predictions_pool, key=lambda x: x['prob'], reverse=True)
        auto_threshold = sorted_all[min(top_n, len(sorted_all)) - 1]['prob']
        print(f"  → 選定閾値（自動）: {auto_threshold*100:.2f}% (上位{top_n}銘柄基準)")
        sorted_p = sorted_all[:top_n]

        print(f"\n[ ◆ 5営業日後 上昇期待 ベストセレクション (v6.0-5d) ◆ ]")
        print(f"  ※ 月曜寄り付きエントリー → 金曜大引けエグジット")
        print(f"  ※ スリッページ{SLIPPAGE*100:.2f}%込み  利益閾値: {PROFIT_THRESHOLD*100:.1f}%超")
        print(f"  ※ アンサンブル{len(ensemble_models)}モデル平均")
        print(f"  {'No.':4} {'銘柄':12} {'上昇確率':>10} {'期待リターン':>12} "
              f"{'モデル分散':>10} {'ケリー配分':>10} {'直近終値':>10}")
        print(f"  {'─'*70}")
        for j, cand in enumerate(sorted_p):
            prob_std_disp = f"σ={cand['prob_std']*100:.1f}%" if cand['prob_std'] > 0 else "  —  "
            kelly_disp    = f"{cand['kelly_pct']*100:.1f}%" if kelly_ok else "  —  "
            exp_ret_disp  = f"{cand['exp_ret']*100:+.2f}%" if cand['exp_ret'] != 0.0 else "  —  "
            print(f"  No.{j+1:2d} | {cand['code']:12} "
                  f"| {cand['prob']*100:8.2f}% "
                  f"| {exp_ret_disp:>10} "
                  f"| {prob_std_disp:>10} "
                  f"| {kelly_disp:>10} "
                  f"| {cand['ref_price']:>9.1f}")

        # ケリー配分サマリー
        if kelly_ok:
            total_kelly = sum(c['kelly_pct'] for c in sorted_p)
            print(f"\n  ポートフォリオ総配分: {total_kelly*100:.1f}%  "
                  f"({'⚠️ 100%超、比例配分を推奨' if total_kelly > 1.0 else '✅ 適切な範囲'})")
            print(f"  ※ ケリー配分は過去{kelly_stats['n_trades']}取引の損益実績に基づきます")
        else:
            print(f"\n  ⚠️ ケリー配分: 蓄積取引数が不足（{kelly_stats['n_trades']}/10件）。"
                  f"  evening を繰り返して実績を積んでください。")

        joblib.dump(sorted_p, PREDICTION_RECORD)
        print(f"\n▶ 予測リストを保存しました。金曜大引け後に evening を実行してください。")

    # ------------------------------------------------------------------
    # アクション6: evening（評価 + 微調整）[v6 改善⑦]
    # ------------------------------------------------------------------
    def evening_evaluate(self):
        print(f"\n===[週次総括] {HORIZON_SHIFT}営業日後パフォーマンス評価 (v6.0-5d) ===")
        if not os.path.exists(PREDICTION_RECORD):
            return print("エラー: 予測ログが見つかりません。morning を先に実行してください。")

        today_recoms = joblib.load(PREDICTION_RECORD)
        returns_list, wins_returns, loses_returns = [], [], []
        wins = losses = 0
        results_sequence = []

        for itm in today_recoms:
            code, ref_p, ref_date = itm['code'], itm['ref_price'], itm['ref_date']
            try:
                tdf = yf.download(code, period='1mo', interval='1d',  # [Fix⑥] 連休対応で余裕を持たせる
                                  progress=False, auto_adjust=True)
                if tdf.empty: continue
                if isinstance(tdf.columns, pd.MultiIndex):
                    tdf.columns = tdf.columns.get_level_values(0)
                tdf.columns = [str(c).strip().capitalize() for c in tdf.columns]

                latest_dt    = str(tdf.index[-1].date())
                days_elapsed = len(pd.bdate_range(ref_date, latest_dt)) - 1

                if days_elapsed < HORIZON_SHIFT:
                    print(f"  {code:10} -> まだ{HORIZON_SHIFT}日経過していません（{days_elapsed}日）")
                    continue

                ref_dt      = pd.to_datetime(ref_date)
                ref_idx     = tdf.index.searchsorted(ref_dt)
                entry_idx   = min(ref_idx + 1, len(tdf) - 1)
                # [v6 改善①] スリッページ込みの実績評価
                entry_price = float(tdf['Open'].iloc[entry_idx]) * (1 + SLIPPAGE)
                exit_price  = float(tdf['Close'].iloc[-1]) * (1 - SLIPPAGE)
                profit_rt   = ((exit_price - entry_price) / entry_price) * 100
                res         = 1 if profit_rt > 0 else 0

                raw_entry = float(tdf['Open'].iloc[entry_idx])
                raw_exit  = float(tdf['Close'].iloc[-1])
                print(f"  {code:10} | {('❌Loss', '✅Win ')[res]}"
                      f" | {profit_rt:+.2f}% (スリッページ込み)"
                      f" | {raw_entry:.1f} → {raw_exit:.1f}"
                      f"  [{days_elapsed}営業日]")

                returns_list.append(profit_rt)
                results_sequence.append(res)
                if res:
                    wins += 1
                    wins_returns.append(profit_rt)
                else:
                    losses += 1
                    loses_returns.append(abs(profit_rt))
            except Exception as e:
                print(f"[WARN] {code}: {e}")

        if not returns_list:
            return print("\n評価対象となった銘柄はありませんでした。")

        total    = wins + losses
        win_rate = wins / total if total > 0 else 0
        avg_ret  = np.mean(returns_list)
        std_ret  = np.std(returns_list) + 1e-9
        avg_win  = np.mean(wins_returns)  if wins_returns  else 0.0
        avg_loss = np.mean(loses_returns) if loses_returns else 0.0
        pf       = sum(wins_returns) / (sum(loses_returns) + 1e-9)
        exp      = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        max_consec = cur = 0
        for r in results_sequence:
            cur        = cur + 1 if r == 0 else 0
            max_consec = max(max_consec, cur)

        sharpe = (avg_ret / std_ret) * np.sqrt(52) if total > 2 else float('nan')
        kelly_b_ratio = avg_win / max(avg_loss, 0.01)

        print(f"\n{'='*60}")
        print(f"  週次スイング サマリー (v6.0-5d)")
        print(f"{'='*60}")
        print(f"  勝率                  : {wins}/{total} ({win_rate*100:.1f}%)")
        print(f"  平均リターン          : {avg_ret:+.2f} % (スリッページ込み)")
        print(f"  平均利益（勝ち）      : +{avg_win:.2f} %")
        print(f"  平均損失（負け）      : -{avg_loss:.2f} %")
        print(f"  損益比 b              : {kelly_b_ratio:.2f}x")
        print(f"  ─────────────────────────────────────────")
        print(f"  期待値(1T当たり)      : {exp:+.2f} %  "
              f"{'✅ プラス期待値' if exp > 0 else '⚠️ マイナス期待値'}")
        print(f"  プロフィットファクター: {pf:.2f}x  "
              f"{'✅ 優秀(2x+)' if pf >= 2.0 else '▶ 良好(1.5x+)' if pf >= 1.5 else '⚠️ 要改善'}")
        print(f"  最大連敗数            : {max_consec} 週")
        print(f"  ボラティリティ        : {std_ret:.2f} %")
        if not np.isnan(sharpe):
            print(f"  年率シャープレシオ    : {sharpe:.2f}")
        print(f"  ※ サンプル数({total}件)が少ないため統計的信頼性は低いです")
        print(f"{'='*60}")

        # [v6 改善⑦] ケリー統計を蓄積保存
        # 既存統計と加重平均（直近データを重視）
        prev_stats = self._load_kelly_stats()
        prev_n     = prev_stats.get('n_trades', 0)
        new_n      = prev_n + total
        if prev_n >= 10:
            # 指数移動平均で更新（過去70% / 今週30%）
            blend = 0.30
            merged_win  = prev_stats['avg_win_pct']  * (1 - blend) + avg_win  * blend
            merged_loss = prev_stats['avg_loss_pct'] * (1 - blend) + avg_loss * blend
            merged_wr   = prev_stats['win_rate']     * (1 - blend) + win_rate * blend
        else:
            merged_win  = avg_win
            merged_loss = avg_loss
            merged_wr   = win_rate
        self._save_kelly_stats(merged_win, merged_loss, merged_wr, new_n)

        # ── モデル微調整は実施しない ──────────────────────────────────
        # 週5〜20件のサンプルは統計的ノイズ。破局的忘却のリスクが収益より大きい。
        # 累積50件以上になったら train_kaggle で正式再訓練すること。
        print(f"\n▶ モデル微調整: 省略（累積{new_n}件 / 推奨50件以上で再訓練）")
        print(f"  ケリー統計のみ更新済み: {KELLY_STATS_FILE}")


# =================================================================
# コマンドライン制御
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RobustDeepTraderAI v6.0-5d",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "mode",
        choices=["download", "preprocess", "train", "train_kaggle", "morning", "evening"],
        help=(
            "download     : 全銘柄CSVをダウンロード\n"
            "preprocess   : チャンク分割npzを生成\n"
            "train        : CSVから直接学習\n"
            "train_kaggle : チャンクから学習（Kaggle対応）\n"
            "morning      : 月曜朝の予測スキャン（全シードアンサンブル）\n"
            "evening      : 金曜大引け後の評価・ケリー統計更新・微調整"
        )
    )
    parser.add_argument("--years",   type=int,  default=5)
    parser.add_argument("--epochs",  type=int,  default=30)
    parser.add_argument("--seq_len", type=int,  default=120)
    parser.add_argument("--batch",   type=int,  default=512)
    parser.add_argument("--top_n",   type=int,  default=5)
    parser.add_argument("--npz",     type=str,  default=None)
    parser.add_argument(
        "--seed", type=int, default=0,
        help=f"アンサンブル用シード番号 0-{N_ENSEMBLE-1} (default: 0)\n"
             f"  3シード全訓練: --seed 0 → --seed 1 → --seed 2 の順に実行"
    )
    parser.add_argument("--reset", action='store_true',
                        help="進捗をリセットして最初から学習")

    args = parser.parse_args()

    if args.reset:
        prog_file = PROGRESS_FILE_TPL.format(args.seed)
        if os.path.exists(prog_file):
            os.remove(prog_file)
            print(f"▶ 進捗リセット: {prog_file} を削除しました")
        else:
            print(f"▶ 進捗ファイルはありませんでした（既にリセット済み）")

    ai = RobustDeepTraderAI(
        seq_len    = args.seq_len,
        epochs     = args.epochs,
        batch_size = args.batch,
        seed       = args.seed,
    )

    if   args.mode == "download":      ai.download_csv_history(years=args.years)
    elif args.mode == "preprocess":    ai.preprocess_to_npz()
    elif args.mode == "train":         ai.train_from_csv()
    elif args.mode == "train_kaggle":  ai.train_from_npz(npz_path=args.npz)
    elif args.mode == "morning":       ai.morning_predict(top_n=args.top_n)
    elif args.mode == "evening":       ai.evening_evaluate()
