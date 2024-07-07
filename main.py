import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import pytesseract
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import threading
import cv2
import numpy as np

# Tesseract 경로 설정 (Windows에서만 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 모델 파일이 저장된 로컬 디렉토리 경로
model_directory = r"C:\Users\gamin\OneDrive\바탕 화면\융탐\gpt2\gpt2_model"

# GPT-2 모델과 토크나이저 로드
try:
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load GPT-2 model.\nError: {str(e)}")

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            if img.width < 500 or img.height < 500:
                img_resized = img
            else:
                img_resized = img.resize((500, 500), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            panel.config(image=img_tk)
            panel.image = img_tk

            # Tesseract OCR로 텍스트 추출
            extracted_text = pytesseract.image_to_string(img, lang='eng')

            # 검은색으로 가려진 부분을 빈칸으로 처리
            processed_text = process_blackened_text(file_path, extracted_text)
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, processed_text)

            # GPT-2로 텍스트 개선
            threading.Thread(target=improve_text_with_gpt2, args=(processed_text,)).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the image.\nError: {str(e)}")

def process_blackened_text(image_path, text):
    # PIL 이미지를 OpenCV 이미지로 변환
    pil_image = Image.open(image_path).convert('L')
    img = np.array(pil_image)

    # 이미지 이진화
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    # 닫힘 연산을 통해 검은색 라인들을 연결
    kernel = np.ones((5, 5), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 검출
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 텍스트에서 검은색으로 가려진 부분 빈칸으로 대체
    blanked_text = ""
    char_index = 0
    for line in text.splitlines():
        for word in line.split():
            if is_word_blackened(mask, word, char_index):
                blanked_text += "[MASK] "
            else:
                blanked_text += word + " "
            char_index += len(word) + 1
        blanked_text += "\n"
        char_index += 1
    return blanked_text

def is_word_blackened(mask, word, char_index):
    # 단어의 위치와 크기를 기반으로 마스크 이미지를 확인하여 검은색으로 가려진 단어인지 판단
    word_length = len(word)
    for i in range(word_length):
        if char_index + i < len(mask.flatten()) and mask.flatten()[char_index + i] == 255:
            return True
    return False

def improve_text_with_gpt2(text):
    try:
        improved_text = ""
        text_segments = re.split(r'(\[MASK\])', text)
        for segment in text_segments:
            if segment == "[MASK]":
                prompt = improved_text.split()[-50:]  # GPT-2 input context length
                input_ids = tokenizer.encode(' '.join(prompt), return_tensors="pt")
                output_ids = model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 1,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    num_beams=5,
                )
                generated_word = tokenizer.decode(output_ids[0], skip_special_tokens=True).split()[-1]
                improved_text += generated_word
            else:
                improved_text += segment

        # GPT-2로 생성된 텍스트를 GUI에 반영
        gpt2_text_box.delete(1.0, tk.END)
        gpt2_text_box.insert(tk.END, improved_text)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to improve the text.\nError: {str(e)}")

def show_help():
    help_window = Toplevel(root)
    help_window.title("프로그램 사용 방법")

    help_text = """
    프로그램 사용 방법:

    1. File 메뉴에서 Open을 클릭하여 이미지를 선택합니다.
    2. 선택한 이미지가 상단에 표시되고, OCR로 추출된 텍스트가 왼쪽 텍스트 박스에 표시됩니다.
    3. OCR로 추출된 텍스트에서 검은색으로 칠해진 부분은 빈칸으로 대체됩니다.
    4. GPT-2가 빈칸을 기반으로 텍스트를 완성하고, 완성된 텍스트는 오른쪽 텍스트 박스에 표시됩니다.
    5. 텍스트 박스에 직접 텍스트를 입력하거나 수정할 수 있습니다.

    각 텍스트 박스에 입력된 내용은 독립적으로 유지됩니다.
    """

    help_label = tk.Label(help_window, text=help_text, justify='left')
    help_label.pack(padx=10, pady=10)

root = tk.Tk()
root.title("Image Viewer with OCR and GPT-2")

# 이미지 패널
panel = tk.Label(root)
panel.grid(row=0, column=0, columnspan=2, sticky="nsew")

# OCR 텍스트 박스
text_box = tk.Text(root, wrap='word', height=10)
text_box.grid(row=1, column=0, sticky="nsew")

# GPT-2 텍스트 박스
gpt2_text_box = tk.Text(root, wrap='word', height=10)
gpt2_text_box.grid(row=1, column=1, sticky="nsew")

# 메뉴
menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_image)
help_menu = tk.Menu(menu)
menu.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="How to use", command=show_help)

# 행과 열 크기 조정
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()
