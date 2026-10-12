"""새 모델용 GUI (result.py 의 2026 버전).

사용법: python app.py
1) [이미지 열기] 로 단어 이미지를 연다
   - 이미지 안의 꽉 찬 검은 상자(가려진 부분)는 자동으로 선택됨
2) 필요하면 마우스로 드래그해서 복원할 부분을 더 표시 (여러 개 가능)
3) [복원] → 오른쪽에 결과와 모델이 읽은 단어가 표시됨
"""
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageFilter, ImageTk

from inpaint.infer import inpaint, load_model

CANVAS_W, CANVAS_H = 500, 200


def detect_black_boxes(gray):
    """꽉 찬 검은 영역(가림 상자)을 찾아 마스크로 반환. 얇은 글자 획은 걸러진다."""
    k = max(3, (gray.height // 12) | 1)  # 홀수 크기 필터. 높이 50 → 5
    dark = gray.point(lambda v: 0 if v < 60 else 255)
    opened = dark.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    return (np.asarray(opened) == 0).astype(np.float32)


class InpaintApp:
    def __init__(self, root, model_path="word_inpainter.pt"):
        self.root = root
        root.title("Word Inpainter 2026")
        self.model = load_model(model_path)

        bar = tk.Frame(root)
        bar.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(bar, text="이미지 열기", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(bar, text="선택 초기화", command=self.clear_mask).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text="복원", command=self.restore).pack(side=tk.LEFT)
        tk.Button(bar, text="결과 저장", command=self.save).pack(side=tk.LEFT, padx=6)
        self.word_var = tk.StringVar(value="모델이 읽은 단어: -")
        tk.Label(bar, textvariable=self.word_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=12)

        body = tk.Frame(root)
        body.pack(padx=10, pady=6)
        self.canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H, bg="white",
                                highlightthickness=1, highlightbackground="#999")
        self.canvas.grid(row=0, column=0)
        self.out_canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H, bg="#eee",
                                    highlightthickness=1, highlightbackground="#999")
        self.out_canvas.grid(row=0, column=1, padx=10)
        tk.Label(body, text="입력 (드래그로 영역 선택)").grid(row=1, column=0)
        tk.Label(body, text="복원 결과").grid(row=1, column=1)

        self.status_var = tk.StringVar(value="[이미지 열기] 를 눌러 단어 이미지를 불러오세요.")
        tk.Label(root, textvariable=self.status_var, anchor="w", fg="#333").pack(fill=tk.X, padx=10, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.image = None
        self.mask = None
        self.result = None
        self.rect = None

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.image = Image.open(path).convert("L")
        except OSError as e:
            messagebox.showerror("오류", f"이미지를 열 수 없습니다:\n{e}")
            return
        w, h = self.image.size
        self.scale = min(CANVAS_W / w, CANVAS_H / h)
        self.disp_size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        self.photo = ImageTk.PhotoImage(self.image.resize(self.disp_size, Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.out_canvas.delete("all")
        self.result = None
        self.word_var.set("모델이 읽은 단어: -")

        self.mask = detect_black_boxes(self.image)
        self.draw_mask_overlay()
        if self.mask.any():
            self.status_var.set("가려진 부분(검은 상자)을 자동으로 찾았어요. [복원] 을 누르세요. "
                                "더 고칠 곳은 드래그로 추가할 수 있어요.")
        else:
            self.status_var.set("복원할 부분을 마우스로 드래그해서 선택한 뒤 [복원] 을 누르세요.")

    def draw_mask_overlay(self):
        """마스크 영역을 빨간 테두리 상자로 표시."""
        self.canvas.delete("sel")
        if self.mask is None or not self.mask.any():
            return
        rows = np.flatnonzero(self.mask.any(1))
        # 연속된 마스크 열 구간마다 하나의 상자로 표시
        cols = self.mask.any(0).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, cols, 0]))
        for x0, x1 in zip(edges[::2], edges[1::2]):
            ys = np.flatnonzero(self.mask[:, x0:x1].any(1))
            y0, y1 = (ys[0], ys[-1] + 1) if len(ys) else (rows[0], rows[-1] + 1)
            s = self.scale
            self.canvas.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s, outline="red", width=2, tags="sel")

    def clear_mask(self):
        if self.image is None:
            return
        self.mask[:] = 0
        self.canvas.delete("sel")
        self.status_var.set("선택을 지웠어요. 복원할 부분을 드래그해서 선택하세요.")

    def on_press(self, e):
        if self.image is None:
            return
        self.start = (e.x, e.y)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2, tags="sel")

    def on_drag(self, e):
        if self.rect:
            self.canvas.coords(self.rect, *self.start, e.x, e.y)

    def on_release(self, e):
        if self.image is None or not self.rect:
            return
        x0, x1 = sorted((self.start[0], e.x))
        y0, y1 = sorted((self.start[1], e.y))
        h, w = self.mask.shape
        x0, x1 = [int(np.clip(v / self.scale, 0, w)) for v in (x0, x1)]
        y0, y1 = [int(np.clip(v / self.scale, 0, h)) for v in (y0, y1)]
        if x1 > x0 and y1 > y0:
            self.mask[y0:y1, x0:x1] = 1
            self.status_var.set("영역을 선택했어요. [복원] 을 누르세요.")
        else:
            self.canvas.delete(self.rect)
        self.rect = None

    def restore(self):
        if self.image is None or self.mask is None or not self.mask.any():
            messagebox.showinfo("안내", "이미지를 열고 복원할 영역을 드래그해서 선택하세요.")
            return
        self.result, word = inpaint(self.model, self.image, self.mask)
        self.word_var.set(f"모델이 읽은 단어: {word}")
        self.out_photo = ImageTk.PhotoImage(self.result.resize(self.disp_size, Image.LANCZOS))
        self.out_canvas.delete("all")
        self.out_canvas.create_image(0, 0, anchor=tk.NW, image=self.out_photo)
        self.status_var.set("복원 완료! [결과 저장] 으로 저장할 수 있어요.")

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.result.save(path)
            self.status_var.set(f"저장했어요: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    InpaintApp(root)
    root.mainloop()
