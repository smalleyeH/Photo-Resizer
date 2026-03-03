import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
import os
import glob
import logging

# ====== 設定區 ======
INPUT_DIR        = "input_photos"      # 輸入資料夾（放原始照片的地方）
OUTPUT_DIR       = "output_slides"     # 輸出資料夾（裁切後的投影片存這裡）
FILE_EXTS        = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]  # 支援的圖片格式
OUTPUT_RATIO     = "16:9"              # 輸出比例："16:9" / "4:3" / "auto"（自動依偵測結果）
OUTPUT_LONG_SIDE = 1920                # 輸出長邊的像素數
GRID_COLS        = 5                   # 縮圖網格每列幾張
THUMB_W          = 240                 # 單張縮圖的寬度（像素）
THUMB_H          = 160                 # 單張縮圖的高度（像素）
VIEWPORT_W       = 1280                # 主視窗寬度
VIEWPORT_H       = 860                 # 主視窗高度
# ====================

# 設定 log 檔，方便事後除錯
logging.basicConfig(
    filename="cropper.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# 預設的輸出尺寸對照表；"auto" 代表依偵測到的四邊形比例自動決定
RATIO_MAP = {"16:9": (1920, 1080), "4:3": (1600, 1200), "auto": None}

PAD    = 8          # 縮圖之間的間距
FOOT_H = 72         # 視窗底部狀態列的高度
IMG_Y0 = 20         # 縮圖內「圖片區域」的起始 Y 座標（上方留給檔名）
IMG_Y1 = THUMB_H - 22  # 縮圖內「圖片區域」的結束 Y 座標（下方留給狀態標籤）

# 每種狀態對應的顏色（BGR 格式，OpenCV 用的是 BGR 不是 RGB）
STATUS_COLOR = {
    "pending": ( 80,  80,  80),   # 灰色：尚未處理
    "auto":    ( 50, 200,  50),   # 綠色：自動偵測成功
    "manual":  ( 50, 180, 240),   # 橙黃：手動編輯過
    "skip":    ( 70,  70,  70),   # 深灰：跳過不處理
    "fail":    ( 40,  40, 220),   # 紅色：自動偵測失敗，需要手動
}

# 狀態文字標籤
STATUS_LABEL = {
    "pending": "...",
    "auto":    "AUTO",
    "manual":  "EDITED",
    "skip":    "SKIP",
    "fail":    "NEED EDIT",
}

# OpenCV 視窗名稱（固定字串，避免中文或特殊字元造成問題）
GRID_WIN   = "Slide Cropper"
EDITOR_WIN = "Point Editor"


# =============================================
# 資料類別：每張圖片的狀態
# =============================================
class ImageState:
    """
    儲存單張圖片的所有資訊：
    - path:     完整檔案路徑
    - filename: 只有檔名（用來顯示）
    - img:      讀進來的原始影像（numpy 陣列）
    - points:   四個角點座標（用來做透視變換裁切）
    - status:   目前狀態（pending/auto/manual/skip/fail）
    - _thumb_cache:  縮圖快取，避免每幀都重畫
    - _thumb_status: 記錄快取是哪個狀態下產生的，狀態變了才需要重畫
    """
    def __init__(self, path: str):
        self.path     = path
        self.filename = os.path.basename(path)
        self.img      = None
        self.points   = None
        self.status   = "pending"
        # === 效能優化：縮圖快取 ===
        # 如果狀態沒變，就不用每次都重新繪製縮圖，大幅減少運算量
        self._thumb_cache  = None   # 快取的縮圖影像
        self._thumb_status = None   # 產生快取時的狀態，用來判斷是否過期

    def load(self) -> bool:
        """讀取圖片檔案，成功回傳 True"""
        self.img = cv2.imread(self.path)
        return self.img is not None

    def invalidate_thumb(self):
        """強制清除縮圖快取（例如手動編輯角點後要呼叫這個）"""
        self._thumb_cache = None
        self._thumb_status = None


# =============================================
# 自動偵測投影片四邊形
# =============================================
def _sort_points(pts):
    """
    把 4 個角點排成固定順序：左上、右上、右下、左下。
    原理：
    - 左上角的 x+y 最小（最靠近原點）
    - 右下角的 x+y 最大（最遠離原點）
    - 右上角的 x-y 最小（x 大但 y 小）
    - 左下角的 x-y 最大（x 小但 y 大）
    """
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)       # 每個點的 x+y
    diff = np.diff(pts, axis=1)  # 每個點的 x-y
    rect[0] = pts[np.argmin(s)]    # 左上
    rect[2] = pts[np.argmax(s)]    # 右下
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下
    return rect


def _find_quad(edge_img, min_area, max_area):
    """
    從邊緣圖中找四邊形候選。
    嘗試多個 approxPolyDP 容差值，增加找到四邊形的機會。
    回傳所有合格候選的 list: [(area, pts), ...]
    """
    cnts = cv2.findContours(edge_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    candidates = []
    for ct in cnts:
        area = cv2.contourArea(ct)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(ct, True)
        for tol in [0.015, 0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(ct, tol * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                rect = _sort_points(pts)
                tl, tr, br, bl = rect
                w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
                h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
                if w < 1 or h < 1:
                    continue
                ratio = max(w, h) / min(w, h)
                if ratio > 4.0:
                    continue
                if not cv2.isContourConvex(approx):
                    continue
                candidates.append((area, rect))
                break
    return candidates


def auto_detect(img):
    """
    Multi-strategy slide detection.

    Strategy A: multi-threshold Canny edge detection
    Strategy B: adaptive threshold (handles uneven lighting)
    Strategy C: Otsu threshold (handles high contrast slides)

    Each strategy generates an edge map, then _find_quad extracts
    quadrilateral candidates. The largest valid candidate wins.
    """
    h0, w0 = img.shape[:2]
    scale = 900 / h0
    view = cv2.resize(img, (int(w0 * scale), 900))
    img_area = view.shape[0] * view.shape[1]
    min_area = img_area * 0.05
    max_area = img_area * 0.98

    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kdilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    all_candidates = []

    # --- Strategy A: multiple Canny thresholds ---
    for lo, hi in [(20, 80), (30, 120), (50, 150), (75, 200)]:
        edged = cv2.Canny(blur, lo, hi)
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kclose)
        edged = cv2.dilate(edged, kdilate, iterations=1)
        all_candidates.extend(_find_quad(edged, min_area, max_area))

    # --- Strategy B: adaptive threshold ---
    adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kclose)
    adapt = cv2.dilate(adapt, kdilate, iterations=1)
    all_candidates.extend(_find_quad(adapt, min_area, max_area))

    # --- Strategy C: Otsu threshold ---
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kclose)
    otsu = cv2.dilate(otsu, kdilate, iterations=1)
    all_candidates.extend(_find_quad(otsu, min_area, max_area))

    if all_candidates:
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        best_pts = all_candidates[0][1]
        return best_pts / scale, True

    h, w = img.shape[:2]
    pts = np.array([[w*.1, h*.1], [w*.9, h*.1],
                    [w*.9, h*.9], [w*.1, h*.9]], dtype="float32")
    return pts, False



# =============================================
# 縮圖繪製器（含快取機制）
# =============================================
def build_thumb(state: ImageState) -> np.ndarray:
    """
    繪製單張圖片的縮圖卡片。

    【效能優化重點】
    原本每一幀都要重畫全部縮圖 -> 現在用快取：
    如果這張圖的狀態沒變，就直接回傳上次畫好的結果。
    """
    # === 快取命中：狀態沒變就直接回傳 ===
    if state._thumb_cache is not None and state._thumb_status == state.status:
        return state._thumb_cache

    col    = STATUS_COLOR[state.status]
    canvas = np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8)

    if state.img is not None:
        h0, w0  = state.img.shape[:2]
        avail_w = THUMB_W - 4
        avail_h = IMG_Y1 - IMG_Y0 - 4
        sc      = min(avail_w / w0, avail_h / h0)
        tw, th  = int(w0 * sc), int(h0 * sc)
        thumb   = cv2.resize(state.img, (tw, th), interpolation=cv2.INTER_AREA)

        # 居中放置
        ox = (THUMB_W - tw) // 2
        oy = IMG_Y0 + (IMG_Y1 - IMG_Y0 - th) // 2
        canvas[oy:oy+th, ox:ox+tw] = thumb

        # 畫偵測到的四邊形
        if state.points is not None and state.status not in ("skip", "pending"):
            pts_c = state.points.copy()
            pts_c[:, 0] = pts_c[:, 0] * sc + ox
            pts_c[:, 1] = pts_c[:, 1] * sc + oy
            pts_i = pts_c.astype(int)
            for i in range(4):
                cv2.line(canvas, tuple(pts_i[i]), tuple(pts_i[(i+1)%4]),
                         col, 1, cv2.LINE_AA)
            for pt in pts_i:
                cv2.circle(canvas, tuple(pt), 3, col, -1, cv2.LINE_AA)

    # 頂部檔名列
    cv2.rectangle(canvas, (0, 0), (THUMB_W, IMG_Y0), (18, 18, 18), -1)
    name = state.filename
    if len(name) > 28:
        name = name[:13] + ".." + name[-11:]
    cv2.putText(canvas, name, (4, IMG_Y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # 底部狀態列
    cv2.rectangle(canvas, (0, IMG_Y1), (THUMB_W, THUMB_H), col, -1)
    cv2.putText(canvas, STATUS_LABEL[state.status], (5, THUMB_H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (THUMB_W-1, THUMB_H-1), col, 2)

    # === 存入快取 ===
    state._thumb_cache  = canvas
    state._thumb_status = state.status
    return canvas


# =============================================
# 角點編輯器
# =============================================
class PointEditor:
    """
    讓使用者用滑鼠拖曳四個角點，手動調整投影片邊界。
    Enter：確認 / Esc：取消 / R：重置 / S：跳過
    """
    SNAP = 20
    # 「角落快捷區」佔圖片寬高的比例（0.4 = 40%）
    # 在左上角 40% 範圍內點擊，即使沒精準點到 TL 角點，
    # 也會自動把 TL 移到點擊位置並開始拖曳
    CORNER_ZONE = 0.4

    def __init__(self, state: ImageState):
        self.state  = state
        h0          = state.img.shape[0]
        self.ratio  = h0 / 900
        self.view   = cv2.resize(state.img,
                                  (int(state.img.shape[1] / self.ratio), 900))
        self.pts    = (state.points / self.ratio).astype("float32")
        self.orig   = self.pts.copy()
        self.sel    = -1
        self.action = "cancel"

    def _corner_zone_index(self, x, y):
        h, w = self.view.shape[:2]
        zw = w * self.CORNER_ZONE
        zh = h * self.CORNER_ZONE
        if   x < zw       and y < zh:       return 0
        elif x > (w - zw) and y < zh:       return 1
        elif x > (w - zw) and y > (h - zh): return 2
        elif x < zw       and y > (h - zh): return 3
        else:                                return -1

    def _corner_zone_index(self, x, y):
        h, w = self.view.shape[:2]
        zw = w * self.CORNER_ZONE
        zh = h * self.CORNER_ZONE
        if   x < zw       and y < zh:       return 0
        elif x > (w - zw) and y < zh:       return 1
        elif x > (w - zw) and y > (h - zh): return 2
        elif x < zw       and y > (h - zh): return 3
        else:                                return -1

    def _draw(self):
        img = self.view.copy()
        pts = self.pts.astype(int)
        h, w = img.shape[:2]
        ov = img.copy()
        cv2.fillPoly(ov, [pts], (0, 255, 0))
        cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)

        # 畫左上角快捷區的淡色提示（讓使用者知道這個區域可以直接點擊設定 TL）
        zone_w = int(w * self.CORNER_ZONE)
        zone_h = int(h * self.CORNER_ZONE)
        zc = (255, 200, 0)
        rects = [
            (0, 0, zone_w, zone_h),
            (w-zone_w, 0, w, zone_h),
            (w-zone_w, h-zone_h, w, h),
            (0, h-zone_h, zone_w, h),
        ]
        zone_ov = img.copy()
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(zone_ov, (x1, y1), (x2, y2), zc, -1)
        cv2.addWeighted(zone_ov, 0.07, img, 0.93, 0, img)
        # L 形角標記，提示這是 TL 快捷區
        m = 40
        corner_marks = [
            [((2,2),(2,min(m,zone_h))), ((2,2),(min(m,zone_w),2))],
            [((w-3,2),(w-3,min(m,zone_h))), ((w-3,2),(max(w-m,w-zone_w),2))],
            [((w-3,h-3),(w-3,max(h-m,h-zone_h))), ((w-3,h-3),(max(w-m,w-zone_w),h-3))],
            [((2,h-3),(2,max(h-m,h-zone_h))), ((2,h-3),(min(m,zone_w),h-3))],
        ]
        for mk in corner_marks:
            for (p1, p2) in mk:
                cv2.line(img, p1, p2, zc, 1, cv2.LINE_AA)

        for i in range(4):
            cv2.line(img, tuple(pts[i]), tuple(pts[(i+1) % 4]),
                     (0, 220, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            c = (0, 0, 255) if i == self.sel else (255, 80, 80)
            cv2.circle(img, tuple(pt), 10, c, -1, cv2.LINE_AA)
            cv2.putText(img, ["TL", "TR", "BR", "BL"][i],
                        (pt[0]+13, pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
        short = (self.state.filename[:30] + "...") \
                if len(self.state.filename) > 30 else self.state.filename
        cv2.putText(img,
                    f"{short}   [Enter]OK [Esc]Cancel [R]Reset [S]Skip"
                    f"  |  Click corner zone to snap point",
                    (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (210, 210, 210), 1, cv2.LINE_AA)
        return img

    def _mouse(self, event, x, y, flags, param):
        """
        滑鼠事件處理（含左上角快捷區功能）：

        點擊邏輯（優先順序）：
        1. 先檢查是否點在任何角點的 SNAP(20px) 範圍內 -> 選中該角點拖曳
        2. 沒點到角點，但在圖片左上角 CORNER_ZONE(40%) 範圍內
           -> 把 TL 角點（index=0）瞬移到點擊位置，並開始拖曳
        3. 都不符合 -> 忽略這次點擊
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 步驟 1：計算點擊位置到每個角點的距離
            dists = [np.linalg.norm(p - [x, y]) for p in self.pts]
            idx   = int(np.argmin(dists))
            if dists[idx] < self.SNAP:
                # 精準點到某個角點，直接選中它
                self.sel = idx
            else:
                # 步驟 2：沒點到角點，檢查是否在左上角快捷區內
                h, w = self.view.shape[:2]
                zone_idx = self._corner_zone_index(x, y)
                if zone_idx >= 0:
                    self.pts[zone_idx] = [float(x), float(y)]
                    self.sel = zone_idx
        elif event == cv2.EVENT_MOUSEMOVE and self.sel != -1:
            h, w = self.view.shape[:2]
            self.pts[self.sel] = [np.clip(x, 0, w-1), np.clip(y, 0, h-1)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.sel = -1

    def run(self):
        cv2.namedWindow(EDITOR_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(EDITOR_WIN, 120, 60)
        cv2.setMouseCallback(EDITOR_WIN, self._mouse)
        while True:
            cv2.imshow(EDITOR_WIN, self._draw())
            key = cv2.waitKey(15) & 0xFF
            if key == 13:
                self.action = "confirm"; break
            elif key == 27:
                self.action = "cancel";  break
            elif key in (ord('r'), ord('R')):
                self.pts = self.orig.copy()
            elif key in (ord('s'), ord('S')):
                self.action = "skip";    break
        cv2.destroyWindow(EDITOR_WIN)
        for _ in range(5):
            cv2.waitKey(1)
        if self.action == "confirm":
            self.state.points = self.pts * self.ratio
            self.state.status = "manual"
            self.state.invalidate_thumb()
        elif self.action == "skip":
            self.state.status = "skip"
            self.state.invalidate_thumb()

_editor_ref = None


# =============================================
# 網格狀態
# =============================================
_states    = []
_scroll_y  = 0
_clicked   = -1
_max_sc    = 0

def _grid_cols():
    return GRID_COLS

def _thumb_xy(idx):
    r, c = divmod(idx, _grid_cols())
    return PAD + c * (THUMB_W + PAD), PAD + r * (THUMB_H + PAD)

def _full_h():
    rows = (len(_states) + _grid_cols() - 1) // _grid_cols()
    return PAD + rows * (THUMB_H + PAD)

def _on_grid_mouse(event, x, y, flags, param):
    global _scroll_y, _clicked, _max_sc
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < VIEWPORT_H - FOOT_H:
            ay = y + _scroll_y
            for i in range(len(_states)):
                gx, gy = _thumb_xy(i)
                if gx <= x < gx + THUMB_W and gy <= ay < gy + THUMB_H:
                    _clicked = i
                    break
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = -60 if flags > 0 else 60
        _scroll_y = int(np.clip(_scroll_y + delta, 0, _max_sc))


def _build_grid_frame(progress_text=""):
    """
    建構主視窗畫面。

    【效能優化 — Viewport Culling（只繪製可見區域）】

    原本：建立涵蓋所有縮圖的巨大畫布 -> 全部畫完 -> 裁切可見區域
    問題：100 張圖每幀都要畫 100 次 resize + 畫線，非常慢

    優化：
    1. 算出哪些「列」在可見範圍內（通常只有 3~5 列）
    2. 只繪製那幾列的縮圖
    3. 搭配 build_thumb() 的快取，狀態沒變的連 resize 都省掉

    結果：捲動時計算量從 O(N) 降到接近 O(1)
    """
    global _max_sc

    fh    = _full_h()
    cols  = _grid_cols()
    fw    = cols * (THUMB_W + PAD) + PAD
    vis_h = VIEWPORT_H - FOOT_H
    _max_sc = max(0, fh - vis_h)

    frame = np.full((VIEWPORT_H, VIEWPORT_W, 3), 35, dtype=np.uint8)
    ox = max(0, (VIEWPORT_W - fw) // 2)

    # === 只繪製可見列的縮圖 ===
    row_h = THUMB_H + PAD
    first_visible_row = max(0, (_scroll_y - PAD) // row_h)
    last_visible_row  = (_scroll_y + vis_h) // row_h
    total_rows = (len(_states) + cols - 1) // cols

    for row in range(first_visible_row, min(last_visible_row + 1, total_rows)):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(_states):
                break

            state = _states[idx]
            gx, gy = _thumb_xy(idx)
            dy = gy - _scroll_y

            if dy + THUMB_H < 0 or dy >= vis_h:
                continue

            thumb = build_thumb(state)

            # 處理邊界裁切（縮圖可能只有一部分在畫面中）
            src_y0 = max(0, -dy)
            src_y1 = min(THUMB_H, vis_h - dy)
            dst_y0 = max(0, dy)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            dx     = gx + ox
            src_x0 = 0
            src_x1 = min(THUMB_W, VIEWPORT_W - dx)
            dst_x0 = dx
            dst_x1 = dst_x0 + (src_x1 - src_x0)

            if src_y1 > src_y0 and src_x1 > src_x0:
                frame[dst_y0:dst_y1, dst_x0:dst_x1] = \
                    thumb[src_y0:src_y1, src_x0:src_x1]

    # 捲軸
    if _max_sc > 0:
        bar_h = max(30, int(vis_h * vis_h / fh))
        bar_y = int(_scroll_y / _max_sc * max(0, vis_h - bar_h))
        cv2.rectangle(frame, (VIEWPORT_W - 9, bar_y),
                      (VIEWPORT_W - 3, bar_y + bar_h), (130, 130, 130), -1)

    # 底部狀態列
    fy = VIEWPORT_H - FOOT_H
    cv2.rectangle(frame, (0, fy), (VIEWPORT_W, VIEWPORT_H), (18, 18, 18), -1)
    cv2.line(frame, (0, fy), (VIEWPORT_W, fy), (60, 60, 60), 1)

    n   = len(_states)
    na  = sum(1 for s in _states if s.status == "auto")
    nm  = sum(1 for s in _states if s.status == "manual")
    ns  = sum(1 for s in _states if s.status == "skip")
    nf  = sum(1 for s in _states if s.status == "fail")
    np_ = sum(1 for s in _states if s.status == "pending")

    stat = (f"Total:{n}  AUTO:{na}  EDITED:{nm}  "
            f"SKIP:{ns}  NEED EDIT:{nf}  Pending:{np_}")
    cv2.putText(frame, stat, (12, fy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (190, 190, 190), 1, cv2.LINE_AA)

    line2 = progress_text if progress_text else \
            "[Click thumbnail] Edit   [W/S or Wheel] Scroll   [Enter] Export all   [Esc] Quit"
    cv2.putText(frame, line2, (12, fy + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (120, 200, 240) if progress_text else (100, 190, 100),
                1, cv2.LINE_AA)

    hint = "[Click thumbnail] Edit   [W/S or Wheel] Scroll   [Enter] Export all   [Esc] Quit"
    if progress_text:
        cv2.putText(frame, hint, (12, fy + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 150, 90), 1, cv2.LINE_AA)

    if nf > 0:
        cv2.putText(frame, f"{nf} need manual edit",
                    (VIEWPORT_W - 220, fy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 240), 1, cv2.LINE_AA)

    return frame


# =============================================
# 匯出功能
# =============================================
def calc_output_size(pts):
    """根據設定計算輸出尺寸"""
    fixed = RATIO_MAP.get(OUTPUT_RATIO)
    if fixed:
        return fixed
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    scale = OUTPUT_LONG_SIDE / max(w, h)
    return int(w * scale), int(h * scale)


def export_all():
    """匯出所有非 skip 的圖片：透視變換拉正 -> 縮放 -> 存檔"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = skipped = errors = 0
    total = len(_states)

    for i, state in enumerate(_states, 1):
        fname = state.filename
        print(f"[{i}/{total}] {fname}", end="  ", flush=True)
        if state.status == "skip":
            print("skipped"); skipped += 1; continue
        try:
            ow, oh = calc_output_size(state.points)
            warped = four_point_transform(state.img, state.points)
            warped = cv2.resize(warped, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
            out    = os.path.join(OUTPUT_DIR, "fixed_" + fname)
            cv2.imwrite(out, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tag    = "AUTO" if state.status == "auto" else "EDIT"
            print(f"[{tag}] -> {out} ({ow}x{oh})")
            saved += 1
            logging.info("%s saved %s", fname, out)
        except Exception as e:
            print(f"ERROR: {e}")
            logging.error("%s %s", fname, e)
            errors += 1

    print(f"\n{'='*50}")
    print(f"Done: {saved} saved  {skipped} skipped  {errors} errors")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")


# =============================================
# 主程式
# =============================================
def main():
    global _states, _clicked, _editor_ref

    paths = []
    for ext in FILE_EXTS:
        paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    paths = sorted(set(paths))

    if not paths:
        print(f"No images found in '{INPUT_DIR}'")
        return

    total = len(paths)
    print(f"Found {total} images.\n")

    _states = [ImageState(p) for p in paths]

    cv2.namedWindow(GRID_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(GRID_WIN, 20, 20)
    cv2.setMouseCallback(GRID_WIN, _on_grid_mouse)
    cv2.imshow(GRID_WIN, _build_grid_frame(f"Detecting... 0 / {total}"))
    cv2.waitKey(1)

    for i, state in enumerate(_states):
        prog = f"Detecting: {i+1} / {total}  -  {state.filename}"
        cv2.imshow(GRID_WIN, _build_grid_frame(prog))
        cv2.waitKey(1)

        if not state.load():
            print(f"  [{i+1}/{total}] Cannot read: {state.filename}")
            state.status = "skip"
        else:
            pts, detected = auto_detect(state.img)
            state.points  = pts
            state.status  = "auto" if detected else "fail"
            tag = "OK  " if detected else "FAIL"
            print(f"  [{i+1}/{total}] {tag} {state.filename}")

        cv2.imshow(GRID_WIN, _build_grid_frame(prog))
        cv2.waitKey(1)

    n_fail = sum(1 for s in _states if s.status == "fail")
    print(f"\nDetection done.  Auto OK: {total - n_fail}  Failed: {n_fail}")
    print("Review thumbnails, click any to edit. Press Enter to export.\n")

    while True:
        cv2.imshow(GRID_WIN, _build_grid_frame())
        key = cv2.waitKey(20) & 0xFF

        if _clicked != -1:
            idx = _clicked
            _clicked = -1
            _editor_ref = PointEditor(_states[idx])
            _editor_ref.run()
            _editor_ref = None

        if key == 13:
            cv2.destroyAllWindows()
            for _ in range(5): cv2.waitKey(1)
            print("\nExporting...\n")
            export_all()
            break
        elif key == 27:
            cv2.destroyAllWindows()
            for _ in range(5): cv2.waitKey(1)
            print("Cancelled.")
            break
        elif key in (ord('w'), ord('W')):
            global _scroll_y
            _scroll_y = max(0, _scroll_y - 80)
        elif key in (ord('s'), ord('S')):
            _scroll_y = min(_max_sc, _scroll_y + 80)


if __name__ == "__main__":
    main()