# fetch_human_ac_wa_rates_only_a_paged_robust.py
# -*- coding: utf-8 -*-
"""
ABC392〜ABC421 の A問題だけ、提出ベース AC率 / WA確率 を収集。
・まずページャから最大ページを取得
・ダメなら page=2 を試し、さらに指数探索→二分探索で最終ページを推定
・最終ページの行数も取得して「総件数 = (last-1)*page_size + last_rows」
・ブロック/ログイン/疑わしいページは debug_html に保存
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

# ===== 設定（ここを編集） =======================================================
ABC_START = 402
ABC_END   = 421
PROBLEM_LETTERS = ["a"]         # A問題のみ
DELAY_SEC = 1.2                 # レートリミット対策（必要なら 1.8〜2.5）
RETRY     = 3
TIMEOUT   = 25
LANG_COOKIE = "en"
USER_AGENT  = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

# ---- 必要ならブラウザの Cookie を注入 ------------------------------------------
USE_BROWSER_COOKIES = True
BROWSER_COOKIE_STRING = "REVEL_SESSION=e7d3589cc0842f6eb201f11979cd90bc0b755fd6-%00_TS%3A1774281703%00%00csrf_token%3AedGshs%2BemvMzxzdPyPXPxBsrwwVlinXtmLy2%2Fb5OhKw%3D%00%00w%3Afalse%00%00SessionKey%3A56e518f76b9b26f1c3da613aa06bbfa50d7d8b0abe9724280f47d5ed71c706d3%00%00UserScreenName%3Akazu107%00%00UserName%3Akazu107%00%00a%3Afalse%00;"  # ← DevTools > Application > Cookies で atcoder.jp の Cookie をコピペ

OUT_CSV   = Path(f"human_rates_only_a_paged_ABC{ABC_START}_ABC{ABC_END}.csv")
DEBUG_DIR = Path("debug_html")
# ==============================================================================


@dataclass
class Counts:
    total: int
    ac: int
    wa: int
    @property
    def ac_rate(self) -> float:
        return (self.ac / self.total) if self.total > 0 else 0.0
    @property
    def wa_prob(self) -> float:
        return (self.wa / self.total) if self.total > 0 else 0.0


# ------------------------ HTTP / HTML ユーティリティ ---------------------------
def _apply_browser_cookies(session: requests.Session):
    if not USE_BROWSER_COOKIES or not BROWSER_COOKIE_STRING.strip():
        return
    jar: Dict[str, str] = {}
    for pair in BROWSER_COOKIE_STRING.split(";"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            jar[k.strip()] = v.strip()
    for k, v in jar.items():
        session.cookies.set(k, v, domain="atcoder.jp")


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    s.cookies.set("language", LANG_COOKIE, domain="atcoder.jp")
    _apply_browser_cookies(s)
    return s


def get(session: requests.Session, url: str, save_tag: Optional[str] = None) -> Optional[str]:
    for attempt in range(1, RETRY + 1):
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        if "login" in r.url:
            _dump_html((save_tag or "login_redirect"), r.text)
            print(f"[block] redirected to login: {r.url}")
            return None
        if r.status_code in (403, 429):
            print(f"[warn] {r.status_code} for {url} (attempt {attempt}/{RETRY})")
            time.sleep(2.0 * attempt)
            continue
        if r.status_code >= 400:
            print(f"[warn] HTTP {r.status_code} for {url}")
            return None

        html = r.text or ""
        if _looks_like_challenge(html):
            _dump_html((save_tag or "challenge"), html)
            print(f"[block] challenge page detected for {url}")
            return None
        return html
    print(f"[warn] failed to fetch after {RETRY} retries: {url}")
    return None


def _looks_like_challenge(html: str) -> bool:
    probes = [
        "cf-chl", "cf_chl", "Attention Required!",
        "Please enable JavaScript", "Access denied", "アクセスが拒否されました"
    ]
    low = html.lower()
    return any(p.lower() in low for p in probes)


def _dump_html(tag: str, html: str):
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        p = DEBUG_DIR / f"{tag}.html"
        p.write_text(html, encoding="utf-8", errors="ignore")
        print(f"[debug] saved HTML → {p}")
    except Exception:
        pass


def extract_rows_and_candidate_maxpage(html: str, save_tag: str) -> Tuple[int, int]:
    """このページの行数、ページャの数字から推測した最大ページ（なければ1）。"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table#submission-table") or soup.select_one("table.table") or soup.find("table")
    if not table:
        _dump_html(save_tag + "_no_table", html)
        print(f"[hint] No submission table found in {save_tag}.html")
        return 0, 1
    rows = table.select("tbody tr")
    this_rows = len(rows)

    # ページャの数字リンク/テキストから最大ページを拾う（href 解析と数字テキストの両方）
    max_page = 1
    pager = soup.select("ul.pagination li, nav ul.pagination li")
    for li in pager:
        # 1) href の ?page= を読む
        a = li.find("a")
        if a and a.get("href"):
            href = a.get("href", "")
            try:
                q = parse_qs(urlparse(href).query)
                p = int(q.get("page", [None])[0]) if q.get("page") else None
                if p and p > max_page:
                    max_page = p
            except Exception:
                pass
        # 2) 数字テキストも読む
        txt = (a.get_text(strip=True) if a else li.get_text(strip=True))
        if txt.isdigit():
            p2 = int(txt)
            if p2 > max_page:
                max_page = p2

    # 0行かつページャがなければ念のため保存
    if this_rows == 0 and max_page == 1:
        _dump_html(save_tag + "_zero_or_blocked", html)

    return this_rows, max_page


def count_total_via_pagination_or_scan(session: requests.Session, base: str, save_prefix: str) -> int:
    """
    1) page=1 を取得し、行数と「候補最大ページ」を読む
    2) 候補最大ページ>1 → その最終ページを読んで合算
    3) 候補最大ページ==1 かつ 行数==20 など「怪しい」→ page=2 を試し、さらに指数→二分探索で最終ページを特定
    """
    # page=1
    url1 = base + "&page=1"
    html1 = get(session, url1, save_tag=f"{save_prefix}_p1")
    if not html1:
        return 0
    rows1, maxpage_guess = extract_rows_and_candidate_maxpage(html1, save_tag=f"{save_prefix}_p1")
    page_size = rows1 if rows1 > 0 else 20

    def _rows_at(page: int) -> int:
        html = get(session, base + f"&page={page}", save_tag=f"{save_prefix}_p{page}")
        if not html:
            return 0
        rows, _ = extract_rows_and_candidate_maxpage(html, save_tag=f"{save_prefix}_p{page}")
        return rows

    # ページャで十分分かる場合
    if maxpage_guess > 1:
        last_rows = _rows_at(maxpage_guess)
        return (maxpage_guess - 1) * page_size + last_rows

    # ここからフォールバック探索
    # 「1ページだけ・行数が page_size（典型:20）」は怪しい → page=2 を確認
    if rows1 >= page_size:
        rows2 = _rows_at(2)
        time.sleep(DELAY_SEC)
        if rows2 == 0:
            # 本当に1ページだった
            return rows1

        # 指数探索で上限を大まかに見つける
        p = 4
        last_nonempty = 2
        while True:
            r = _rows_at(p)
            time.sleep(DELAY_SEC)
            if r == 0:
                # 上限を超えた
                hi = p
                lo = last_nonempty
                break
            last_nonempty = p
            p *= 2
            if p > 20000:  # 安全弁
                break

        # 二分探索：lo(≠0) と hi(=0) の間で最後の非ゼロページを探す
        def _binary_search_last_nonempty(lo: int, hi: int) -> Tuple[int, int]:
            # 戻り値: (last_page, last_rows)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                r = _rows_at(mid)
                time.sleep(DELAY_SEC)
                if r > 0:
                    lo = mid
                else:
                    hi = mid
            last_rows = _rows_at(lo)
            return lo, last_rows

        # p が極端に大きくなった場合のガード
        if rows2 > 0 and last_nonempty >= 2:
            last_page, last_rows = _binary_search_last_nonempty(last_nonempty, hi)
            return (last_page - 1) * page_size + last_rows

    # rows1 < page_size（本当に小規模）か、探索不能
    return rows1


def count_total_for_filter(session: requests.Session, contest: str, problem_id: str, status: Optional[str]) -> int:
    # 英語UI・作成順固定で安定化
    base = (
        f"https://atcoder.jp/contests/{contest}/submissions"
        f"?lang=en&order_by=created&f.Task={problem_id}"
    )
    if status:
        base += f"&f.Status={status}"
    return count_total_via_pagination_or_scan(session, base, save_prefix=f"{contest}_{problem_id}_{status or 'ALL'}")


def list_existing_problems(session: requests.Session, contest_num: int) -> List[str]:
    contest = f"abc{contest_num}"
    url = f"https://atcoder.jp/contests/{contest}/tasks?lang=en"
    html = get(session, url, save_tag=f"{contest}_tasks")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    ids = []
    for a in soup.select("a[href*='/tasks/']"):
        href = a.get("href", "")
        if "/tasks/abc" in href:
            pid = href.split("/tasks/")[-1].strip("/")
            if pid.startswith("abc"):
                ids.append(pid)
    return sorted(set(ids))


def filter_by_letters(problem_ids: List[str], contest_num: int, letters: Optional[List[str]]) -> List[str]:
    if not letters:
        return problem_ids
    letters = [s.lower() for s in letters]
    head = f"abc{contest_num}_"
    return [pid for pid in problem_ids
            if pid.startswith(head) and pid.split("_")[-1].lower() in letters]


def safe_counts(session: requests.Session, contest: str, problem_id: str) -> Counts:
    total = count_total_for_filter(session, contest, problem_id, status=None)
    time.sleep(DELAY_SEC)
    ac    = count_total_for_filter(session, contest, problem_id, status="AC")
    time.sleep(DELAY_SEC)
    wa    = count_total_for_filter(session, contest, problem_id, status="WA")
    time.sleep(DELAY_SEC)
    return Counts(total=total, ac=ac, wa=wa)


# --------------------------------- main ---------------------------------------
def main():
    session = new_session()

    rows = ["contest,problem_id,total_submissions,ac_submissions,wa_submissions,ac_rate,wa_prob"]
    for num in range(ABC_START, ABC_END + 1):
        contest = f"abc{num}"
        pids_all = list_existing_problems(session, num)
        pids = filter_by_letters(pids_all, num, PROBLEM_LETTERS)
        if not pids:
            print(f"[warn] {contest}: A問題が見つかりません（取得失敗/ブロックの可能性）。")
            continue

        for pid in pids:
            c = safe_counts(session, contest, pid)
            rows.append(f"{contest},{pid},{c.total},{c.ac},{c.wa},{c.ac_rate:.6f},{c.wa_prob:.6f}")
            print(f"[ok] {contest}/{pid}: total={c.total} AC={c.ac} WA={c.wa} "
                  f"(ACrate={c.ac_rate:.3%}, WApr={c.wa_prob:.3%})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text("\n".join(rows), encoding="utf-8")
    print(f"[save] {OUT_CSV}")

if __name__ == "__main__":
    main()
