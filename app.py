"""새 모델용 GUI (result.py 의 2026 버전).

사용법: python app.py
1) [이미지 열기] 로 단어 이미지를 연다
2) 마우스로 드래그해서 가려진/복원할 부분을 표시 (여러 개 가능)
3) [복원] → 오른쪽에 결과와 모델이 읽은 단어가 표시됨
"""
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from inpaint.infer import inpaint, load_model

CANVAS_W, CANVAS_H = 500, 200


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
        self.canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H, bg="white")
        self.canvas.grid(row=0, column=0)
        self.out_label = tk.Label(body, width=CANVAS_W, height=CANVAS_H, bg="#eee")
        self.out_label.grid(row=0, column=1, padx=10)
        tk.Label(body, text="입력 (드래그로 영역 선택)").grid(row=1, column=0)
        tk.Label(body, text="복원 결과").grid(row=1, column=1)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.image = None
        self.result = None
        self.rect = None

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self.image = Image.open(path).convert("L")
        w, h = self.image.size
        self.scale = min(CANVAS_W / w, CANVAS_H / h)
        self.disp_size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        self.photo = ImageTk.PhotoImage(self.image.resize(self.disp_size))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.mask = np.zeros((h, w), dtype=np.float32)
        self.word_var.set("모델이 읽은 단어: -")

    def clear_mask(self):
        if self.image is None:
            return
        self.mask[:] = 0
        self.canvas.delete("sel")

    def on_press(self, e):
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
            self.canvas.itemconfig(self.rect, fill="black", stipple="gray50")
        else:
            self.canvas.delete(self.rect)
        self.rect = None

    def restore(self):
        if self.image is None or not self.mask.any():
            messagebox.showinfo("안내", "이미지를 열고 복원할 영역을 드래그해서 선택하세요.")
            return
        self.result, word = inpaint(self.model, self.image, self.mask)
        self.word_var.set(f"모델이 읽은 단어: {word}")
        self.out_photo = ImageTk.PhotoImage(self.result.resize(self.disp_size))
        self.out_label.config(image=self.out_photo, width=self.disp_size[0], height=self.disp_size[1])

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.result.save(path)


if __name__ == "__main__":
    root = tk.Tk()
    InpaintApp(root)
    root.mainloop()
