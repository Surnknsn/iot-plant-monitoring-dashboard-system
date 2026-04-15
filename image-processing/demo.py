# -*- coding: utf-8 -*-
import os, time, argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utils ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_histogram(gray, out_path, title="Histogram"):
    plt.figure()
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Preprocess ----------
def auto_contrast_linear(gray):
    rmin = float(np.min(gray))
    rmax = float(np.max(gray))
    if rmax - rmin < 1e-6:
        return gray.copy()
    stretched = (gray.astype(np.float32) - rmin) * 255.0 / (rmax - rmin)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def equalize_gray(gray):
    return cv2.equalizeHist(gray)

# ---------- Thresholding ----------
def do_thresholds(gray, blur_ksize=5, manual_thresh=127):
    if blur_ksize > 0:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    results = {}
    _, bin_      = cv2.threshold(gray_blur, manual_thresh, 255, cv2.THRESH_BINARY)
    _, bin_inv   = cv2.threshold(gray_blur, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    _, trunc     = cv2.threshold(gray_blur, manual_thresh, 255, cv2.THRESH_TRUNC)
    _, tozero    = cv2.threshold(gray_blur, manual_thresh, 255, cv2.THRESH_TOZERO)
    _, tozeroinv = cv2.threshold(gray_blur, manual_thresh, 255, cv2.THRESH_TOZERO_INV)
    _, otsu      = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results["binary"]      = bin_
    results["binary_inv"]  = bin_inv
    results["trunc"]       = trunc
    results["tozero"]      = tozero
    results["tozero_inv"]  = tozeroinv
    results["otsu_binary"] = otsu

    return results, gray_blur

# ---------- Postprocess & Measure ----------
def morph_process(mask, k_open=3, k_close=3):
    m = mask.copy()
    if k_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    return m

def measure_and_annotate(bgr, mask, area_min=50):
    H, W = mask.shape[:2]
    area_ratio = 100.0 * (np.count_nonzero(mask) / float(H * W))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = bgr.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        count += 1
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, f"#{count}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0,255,0), 2, cv2.LINE_AA)

    cv2.putText(vis, f"Objects: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,200,255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Area%: {area_ratio:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,200,255), 2, cv2.LINE_AA)
    return vis, count, area_ratio

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="plant.jpg", help="ชื่อไฟล์รูป (เช่น plant.jpg)")
    ap.add_argument("--thresh", type=int, default=127, help="ค่า threshold (ใช้กับโหมดคงที่)")
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (เลขคี่ เช่น 3/5/7)")
    ap.add_argument("--open", type=int, default=3, help="kernel เปิดรู (open)")
    ap.add_argument("--close", type=int, default=3, help="kernel ปิดรู (close)")
    args = ap.parse_args()

    bgr = cv2.imread(args.src)
    assert bgr is not None, f"อ่านรูปไม่ได้: {args.src}"

    out_dir = os.path.join("outputs", timestamp())
    ensure_dir(out_dir)

    # ต้นฉบับ
    cv2.imwrite(os.path.join(out_dir, "00_original.jpg"), bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    save_histogram(gray, os.path.join(out_dir, "00_hist_original.png"), "Histogram (original)")

    # Auto-contrast + Equalization
    gray_auto = auto_contrast_linear(gray)
    gray_eq   = equalize_gray(gray)
    cv2.imwrite(os.path.join(out_dir, "02_gray_auto_contrast.jpg"), gray_auto)
    cv2.imwrite(os.path.join(out_dir, "03_gray_equalized.jpg"), gray_eq)

    # Threshold
    th_results, gray_blur = do_thresholds(gray_eq, blur_ksize=args.blur, manual_thresh=args.thresh)
    cv2.imwrite(os.path.join(out_dir, "04_gray_blur.jpg"), gray_blur)

    # ผลลัพธ์แต่ละโหมด
    for name, mask in th_results.items():
        mask_m = morph_process(mask, k_open=args.open, k_close=args.close)
        vis, cnt, area = measure_and_annotate(bgr, mask_m, area_min=50)
        cv2.imwrite(os.path.join(out_dir, f"vis_{name}.jpg"), vis)

    print(f"\n✅ ประมวลผลเสร็จสิ้น! รูปถูกบันทึกในโฟลเดอร์: {out_dir}")

if __name__ == "__main__":
    main()
