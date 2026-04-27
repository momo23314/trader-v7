"""
notify_discord.py  ― RobustDeepTraderAI v6.0-5d 向け Discord 通知スクリプト
===========================================================================

呼び出し方（GitHub Actions / ローカル共通）:
  python notify_discord.py saturday      # 土曜: 銘柄予測通知
  python notify_discord.py weekday       # 月〜木: 日次変動追跡通知
  python notify_discord.py friday        # 金曜: 週次総括通知
  python notify_discord.py monday_mode   # 月曜祝日代替: 起点始値記録
  python notify_discord.py friday_mode   # 金曜祝日代替: 週次総括（2重実行防止付き）

環境変数:
  DISCORD_WEBHOOK_URL  Discord の Webhook URL（GitHub Secret から渡す）

依存:
  pip install yfinance pandas numpy joblib requests jpholiday
"""

import os
import sys
import json
import datetime
import traceback
from pathlib import Path

import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import jpholiday
    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False

# =================================================================
# 設定
# =================================================================
WEBHOOK_URL          = os.environ.get("DISCORD_WEBHOOK_URL", "")
PREDICTION_FILE      = "latest_predictions_5d.pkl"
MONDAY_PRICES_FILE   = "monday_open_prices.json"
CUMULATIVE_STATS_FILE = "cumulative_stats.json"
KELLY_STATS_FILE     = "kelly_stats_v6.json"
HORIZON_SHIFT        = 5          # 月曜寄り付き → 金曜大引け（5営業日）
SLIPPAGE             = 0.0015     # 0.15% (v6と統一)
MAX_EMBED_FIELDS     = 25         # Discord embed フィールド上限

# =================================================================
# ユーティリティ
# =================================================================
def today_jst() -> datetime.date:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()

def is_jp_holiday(d: datetime.date) -> bool:
    if not HAS_JPHOLIDAY:
        return False
    return jpholiday.is_holiday(d) or d.weekday() >= 5

def send_discord(content: str = "", embeds: list = None, username: str = "TraderBot v6"):
    """Discord Webhook に送信。embeds は最大10件ずつ分割して送る。"""
    if not WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL が設定されていません。標準出力に出力します。")
        print(content)
        if embeds:
            for e in embeds:
                print(json.dumps(e, ensure_ascii=False, indent=2))
        return

    # embeds は Discord の制限（1リクエスト10件）に合わせて分割
    embed_chunks = [embeds[i:i+10] for i in range(0, max(len(embeds), 1), 10)] if embeds else [[]]
    for i, chunk in enumerate(embed_chunks):
        payload = {"username": username}
        if i == 0 and content:
            payload["content"] = content
        if chunk:
            payload["embeds"] = chunk
        try:
            r = requests.post(WEBHOOK_URL, json=payload, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Discord送信失敗: {e}")

def load_json(path: str, default=None):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default if default is not None else {}

def save_json(path: str, data: dict):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_predictions() -> list:
    if not Path(PREDICTION_FILE).exists():
        return []
    try:
        return joblib.load(PREDICTION_FILE)
    except Exception:
        return []

def fetch_current_price(code: str) -> tuple[float | None, float | None]:
    """(現在値 or 直近終値, 始値) を返す。失敗時は (None, None)。"""
    try:
        df = yf.download(code, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        close = float(df["Close"].dropna().iloc[-1])
        open_ = float(df["Open"].dropna().iloc[-1])
        return close, open_
    except Exception:
        return None, None

def load_kelly_stats() -> dict:
    return load_json(KELLY_STATS_FILE, {
        "avg_win_pct": 2.0, "avg_loss_pct": 2.0,
        "win_rate": 0.50, "n_trades": 0, "updated_at": "—"
    })

def kelly_fraction(prob: float, avg_win: float, avg_loss: float,
                   km: float = 0.5, cap: float = 0.30) -> float:
    if prob <= 0.5 or avg_loss <= 0:
        return 0.0
    b = avg_win / max(avg_loss, 0.1)
    q = 1.0 - prob
    return float(np.clip((b * prob - q) / b * km, 0.0, cap))

def color_from_ret(ret_pct: float) -> int:
    """リターン値からDiscord embed カラーコードを返す。"""
    if ret_pct >= 2.0:   return 0x2ECC71   # 緑（大幅上昇）
    if ret_pct >= 0.0:   return 0x57F287   # 薄緑
    if ret_pct >= -2.0:  return 0xE67E22   # オレンジ
    return 0xED4245                         # 赤（大幅下落）


# =================================================================
# MODE: saturday — 銘柄予測通知
# =================================================================
def run_saturday():
    preds = load_predictions()
    today = today_jst()
    kelly = load_kelly_stats()
    kelly_ok = kelly["n_trades"] >= 10

    if not preds:
        send_discord(
            content="⚠️ 予測ファイルが見つかりません。`morning` を先に実行してください。"
        )
        return

    # ─ ヘッダー embed ─
    header = {
        "title": "📊 今週の注目銘柄 (v6.0-5d アンサンブル)",
        "description": (
            f"**予測日**: {today.strftime('%Y年%m月%d日 (%a)')}\n"
            f"**戦略**: 月曜寄り付きエントリー → 金曜大引けエグジット（5営業日）\n"
            f"**スリッページ込み** | **3シードアンサンブル平均**"
        ),
        "color": 0x5865F2,
        "footer": {"text": "ケリー配分は過去実績に基づく参考値です。投資は自己責任で。"},
    }

    # ─ 銘柄別 embed ─
    fields_embed_list = []
    for i, p in enumerate(preds, 1):
        prob     = p.get("prob", 0.0)
        exp_ret  = p.get("exp_ret", 0.0)
        prob_std = p.get("prob_std", 0.0)
        ref_p    = p.get("ref_price", 0.0)
        code     = p.get("code", "—")

        kf = kelly_fraction(prob, kelly["avg_win_pct"], kelly["avg_loss_pct"]) if kelly_ok else None

        std_str   = f"σ={prob_std*100:.1f}%" if prob_std > 0 else "—"
        ret_str   = f"{exp_ret*100:+.2f}%" if exp_ret != 0.0 else "—"
        kelly_str = f"{kf*100:.1f}%" if kf is not None else "—(実績不足)"

        fields_embed_list.append({
            "title": f"No.{i}  {code}",
            "description": (
                f"**上昇確率**: `{prob*100:.2f}%`\n"
                f"**期待リターン**: `{ret_str}`\n"
                f"モデル分散: `{std_str}`  直近終値: `{ref_p:,.0f}`\n"
                f"ケリー配分: `{kelly_str}`"
            ),
            "color": 0x2ECC71 if prob >= 0.65 else 0x57F287 if prob >= 0.55 else 0xFEE75C,
        })

    # ─ ケリー統計 embed ─
    kelly_embed = {
        "title": "📐 ケリー基準 — 現在の実績統計",
        "color": 0x99AAB5,
        "fields": [
            {"name": "累積取引数",  "value": str(kelly["n_trades"]),               "inline": True},
            {"name": "実績勝率",    "value": f"{kelly['win_rate']*100:.1f}%",       "inline": True},
            {"name": "損益比",      "value": f"{kelly['avg_win_pct']/max(kelly['avg_loss_pct'],0.01):.2f}x", "inline": True},
            {"name": "平均利益",    "value": f"+{kelly['avg_win_pct']:.2f}%",       "inline": True},
            {"name": "平均損失",    "value": f"-{kelly['avg_loss_pct']:.2f}%",      "inline": True},
            {"name": "最終更新",    "value": kelly["updated_at"],                   "inline": True},
        ],
        "footer": {"text": "10件以上で配分計算が有効になります"},
    }

    send_discord(
        content=f"🔔 **{today.strftime('%m/%d')} 週の銘柄予測** — {len(preds)}銘柄を選定",
        embeds=[header] + fields_embed_list + [kelly_embed]
    )
    print(f"✅ saturday 通知完了 ({len(preds)}銘柄)")


# =================================================================
# MODE: weekday — 日次変動追跡通知
# =================================================================
def run_weekday():
    today      = today_jst()
    dow        = today.weekday()   # 0=月, 4=金
    preds      = load_predictions()
    mon_prices = load_json(MONDAY_PRICES_FILE, {})

    if is_jp_holiday(today):
        send_discord(content=f"🎌 本日({today})は日本の祝日のためスキップします。")
        return

    # ── 月曜日: 起点始値を記録 ──
    if dow == 0:
        if not preds:
            send_discord(content="⚠️ 予測ファイルが見つかりません。`morning` を先に実行してください。")
            return

        new_prices = {}
        fields = []
        for p in preds:
            code = p["code"]
            _, open_p = fetch_current_price(code)
            if open_p:
                entry = open_p * (1 + SLIPPAGE)
                new_prices[code] = {
                    "open_price"  : open_p,
                    "entry_price" : entry,   # スリッページ込み
                    "record_date" : str(today),
                    "ref_price"   : p.get("ref_price", 0.0),
                    "prob"        : p.get("prob", 0.0),
                    "exp_ret"     : p.get("exp_ret", 0.0),
                }
                fields.append({
                    "name"  : code,
                    "value" : f"始値 `{open_p:,.0f}` → エントリー価格 `{entry:,.1f}`",
                    "inline": True,
                })

        save_json(MONDAY_PRICES_FILE, new_prices)

        embed = {
            "title"      : f"🔔 月曜寄り付き — 起点価格を記録 ({today})",
            "description": f"{len(new_prices)}銘柄のエントリー価格（スリッページ{SLIPPAGE*100:.2f}%込み）を記録しました。",
            "color"      : 0x5865F2,
            "fields"     : fields[:MAX_EMBED_FIELDS],
        }
        send_discord(embeds=[embed])
        print(f"✅ monday 起点価格記録 ({len(new_prices)}銘柄)")
        return

    # ── 火〜木: 日次変動を追跡 ──
    if not mon_prices:
        send_discord(content="⚠️ 月曜の起点価格が未記録です。")
        return

    day_names = {1:"火", 2:"水", 3:"木"}
    day_str   = day_names.get(dow, str(dow))
    fields    = []
    total_ret = []

    for code, rec in mon_prices.items():
        curr, _ = fetch_current_price(code)
        if curr is None:
            continue
        entry   = rec.get("entry_price", rec.get("open_price", 1))
        ret_pct = (curr - entry) / entry * 100
        total_ret.append(ret_pct)
        sign    = "🟢" if ret_pct >= 0 else "🔴"
        fields.append({
            "name"  : f"{sign} {code}",
            "value" : f"現在値 `{curr:,.0f}`  変動 `{ret_pct:+.2f}%`",
            "inline": True,
        })

    avg_ret = np.mean(total_ret) if total_ret else 0.0
    embed = {
        "title"  : f"📈 {day_str}曜日 日次変動レポート ({today})",
        "description": f"ポートフォリオ平均変動: **`{avg_ret:+.2f}%`**（エントリーからの累積）",
        "color"  : color_from_ret(avg_ret),
        "fields" : fields[:MAX_EMBED_FIELDS],
    }
    send_discord(embeds=[embed])
    print(f"✅ weekday 通知完了 (平均 {avg_ret:+.2f}%)")


# =================================================================
# MODE: friday — 週次総括
# =================================================================
def run_friday(guard_double_run: bool = False):
    today      = today_jst()
    mon_prices = load_json(MONDAY_PRICES_FILE, {})
    cum        = load_json(CUMULATIVE_STATS_FILE, {
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_profit_pct": 0.0, "history": []
    })

    if is_jp_holiday(today):
        send_discord(content=f"🎌 本日({today})は日本の祝日のためスキップします。")
        return

    # 2重実行防止
    if guard_double_run:
        last_run = cum.get("last_friday_run", "")
        if last_run == str(today):
            send_discord(content=f"⚠️ 本日({today})はすでに週次総括を実行済みです（2重実行防止）。")
            return

    if not mon_prices:
        send_discord(content="⚠️ 月曜の起点価格が未記録です。週次総括をスキップします。")
        return

    # ── 実績集計 ──
    results  = []
    win_rets = []
    los_rets = []

    for code, rec in mon_prices.items():
        curr, _ = fetch_current_price(code)
        if curr is None:
            continue
        entry   = rec.get("entry_price", rec.get("open_price", 1))
        # スリッページ込み exit
        exit_p  = curr * (1 - SLIPPAGE)
        ret_pct = (exit_p - entry) / entry * 100
        won     = ret_pct > 0

        results.append({
            "code"    : code,
            "entry"   : entry,
            "exit"    : exit_p,
            "ret_pct" : ret_pct,
            "won"     : won,
            "prob"    : rec.get("prob", 0.0),
            "exp_ret" : rec.get("exp_ret", 0.0),
        })
        if won: win_rets.append(ret_pct)
        else:   los_rets.append(abs(ret_pct))

    if not results:
        send_discord(content="⚠️ 今週の評価対象銘柄がありませんでした。")
        return

    # ── 統計計算 ──
    wins      = sum(1 for r in results if r["won"])
    losses    = len(results) - wins
    win_rate  = wins / len(results)
    avg_ret   = np.mean([r["ret_pct"] for r in results])
    avg_win   = np.mean(win_rets)  if win_rets  else 0.0
    avg_loss  = np.mean(los_rets) if los_rets else 0.0
    pf        = sum(win_rets) / (sum(los_rets) + 1e-9)
    exp_val   = win_rate * avg_win - (1 - win_rate) * avg_loss

    # ── 累計統計を更新 ──
    cum["total_trades"]    += len(results)
    cum["wins"]            += wins
    cum["losses"]          += losses
    cum["total_profit_pct"] = cum.get("total_profit_pct", 0.0) + avg_ret
    cum["last_friday_run"]  = str(today)
    cum.setdefault("history", []).append({
        "date"    : str(today),
        "trades"  : len(results),
        "wins"    : wins,
        "win_rate": round(win_rate, 4),
        "avg_ret" : round(avg_ret, 4),
        "pf"      : round(pf, 4),
    })
    save_json(CUMULATIVE_STATS_FILE, cum)
    # 月曜価格ファイルをクリア（次週のために）
    save_json(MONDAY_PRICES_FILE, {})

    # ── 累計勝率 ──
    cum_wr = cum["wins"] / max(cum["total_trades"], 1)

    # ── embeds 構築 ──
    # 1. サマリー
    summary_embed = {
        "title"      : f"🏁 週次総括レポート ({today})",
        "description": (
            f"**今週の結果**: {wins}勝 {losses}敗 / 勝率 `{win_rate*100:.1f}%`\n"
            f"**平均リターン**: `{avg_ret:+.2f}%` (スリッページ両側{SLIPPAGE*100:.2f}%込み)\n"
            f"**プロフィットファクター**: `{pf:.2f}x`  |  **期待値**: `{exp_val:+.2f}%`"
        ),
        "color"  : color_from_ret(avg_ret),
        "fields" : [
            {"name": "平均利益 (勝ち)",  "value": f"+{avg_win:.2f}%",   "inline": True},
            {"name": "平均損失 (負け)",  "value": f"-{avg_loss:.2f}%",  "inline": True},
            {"name": "損益比",           "value": f"{avg_win/max(avg_loss,0.01):.2f}x", "inline": True},
        ],
    }

    # 2. 個別銘柄
    detail_fields = []
    for r in sorted(results, key=lambda x: x["ret_pct"], reverse=True):
        icon = "✅" if r["won"] else "❌"
        detail_fields.append({
            "name"  : f"{icon} {r['code']}",
            "value" : (f"予測 `{r['prob']*100:.1f}%` | "
                       f"実績 `{r['ret_pct']:+.2f}%` "
                       f"({r['entry']:,.0f}→{r['exit']:,.0f})"),
            "inline": False,
        })
    detail_embed = {
        "title"  : "📋 銘柄別 実績一覧",
        "color"  : 0x99AAB5,
        "fields" : detail_fields[:MAX_EMBED_FIELDS],
    }

    # 3. 累計統計
    cum_embed = {
        "title"  : "📊 累計パフォーマンス",
        "color"  : 0x5865F2,
        "fields" : [
            {"name": "累計取引数",    "value": str(cum["total_trades"]),         "inline": True},
            {"name": "累計勝率",      "value": f"{cum_wr*100:.1f}%",             "inline": True},
            {"name": "累計勝ち",      "value": f"{cum['wins']}回",               "inline": True},
            {"name": "累計負け",      "value": f"{cum['losses']}回",             "inline": True},
            {"name": "累計平均Ret",   "value": f"{cum['total_profit_pct']/max(len(cum['history']),1):+.2f}% / 週", "inline": True},
            {"name": "記録週数",      "value": f"{len(cum['history'])}週",       "inline": True},
        ],
        "footer": {"text": f"v6.0-5d | スリッページ{SLIPPAGE*100:.2f}%込み"},
    }

    send_discord(
        content=f"🏁 **週次総括** ({today})",
        embeds=[summary_embed, detail_embed, cum_embed]
    )
    print(f"✅ friday 通知完了 ({wins}勝{losses}敗, 平均{avg_ret:+.2f}%)")


# =================================================================
# MODE: monday_mode — 月曜祝日代替（起点記録のみ）
# =================================================================
def run_monday_mode():
    """月曜が祝日の場合、翌営業日に手動実行して起点価格を記録する。"""
    today = today_jst()
    preds = load_predictions()
    if not preds:
        send_discord(content="⚠️ 予測ファイルが見つかりません。")
        return

    new_prices = {}
    fields     = []
    for p in preds:
        code = p["code"]
        _, open_p = fetch_current_price(code)
        if open_p:
            entry = open_p * (1 + SLIPPAGE)
            new_prices[code] = {
                "open_price"  : open_p,
                "entry_price" : entry,
                "record_date" : str(today),
                "ref_price"   : p.get("ref_price", 0.0),
                "prob"        : p.get("prob", 0.0),
                "exp_ret"     : p.get("exp_ret", 0.0),
                "note"        : "monday_mode (祝日代替)",
            }
            fields.append({
                "name"  : code,
                "value" : f"始値 `{open_p:,.0f}` → エントリー `{entry:,.1f}`",
                "inline": True,
            })

    save_json(MONDAY_PRICES_FILE, new_prices)
    embed = {
        "title"      : f"🔔【祝日代替】起点価格を記録 ({today})",
        "description": f"月曜祝日のため本日を週の起点とします。{len(new_prices)}銘柄を記録。",
        "color"      : 0xFEE75C,
        "fields"     : fields[:MAX_EMBED_FIELDS],
    }
    send_discord(embeds=[embed])
    print(f"✅ monday_mode 完了 ({len(new_prices)}銘柄)")


# =================================================================
# MODE: friday_mode — 金曜祝日代替（2重実行防止付き）
# =================================================================
def run_friday_mode():
    """金曜が祝日の場合、前営業日（木曜）に手動実行して週次総括を行う。"""
    run_friday(guard_double_run=True)
    print("✅ friday_mode 完了")


# =================================================================
# エントリーポイント
# =================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python notify_discord.py [saturday|weekday|friday|monday_mode|friday_mode]")
        sys.exit(1)

    mode = sys.argv[1].strip().lower()
    print(f"▶ notify_discord.py 実行: mode={mode}  日付={today_jst()}")

    try:
        if   mode == "saturday"   : run_saturday()
        elif mode == "weekday"    : run_weekday()
        elif mode == "friday"     : run_friday(guard_double_run=False)
        elif mode == "monday_mode": run_monday_mode()
        elif mode == "friday_mode": run_friday_mode()
        else:
            print(f"[ERROR] 未知のmode: {mode}")
            sys.exit(1)
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] 例外発生:\n{tb}")
        send_discord(content=f"🚨 **TraderBot エラー** (`{mode}`)\n```\n{tb[:1800]}\n```")
        sys.exit(1)
