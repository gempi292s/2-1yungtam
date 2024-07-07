import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.models import Model

# 이미지 폴더 경로
original_folder = "word image folder"
masked_folder = "word image folder hidden"

# 이미지 크기
img_height, img_width = 50, 200

# 이미지 불러오기 및 전처리 함수
def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, color_mode='grayscale', target_size=(img_height, img_width))
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

# 원본 이미지와 가려진 이미지 로드
original_images = load_images(original_folder)
masked_images = load_images(masked_folder)

# 모델 정의
input_img = Input(shape=(img_height, img_width, 1))

# 인코더
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 디코더
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# 크기 조정
decoded = Cropping2D(((1, 1), (0, 0)))(decoded)  # 위아래로 1픽셀씩 크롭

# 모델 구성
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습
autoencoder.fit(masked_images, original_images,
                epochs=5,
                batch_size=16,
                shuffle=True)

# 학습된 모델을 사용하여 복원 예측
decoded_imgs = autoencoder.predict(masked_images)

# 결과 이미지 저장
result_folder = "result_images"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

for i, decoded_img in enumerate(decoded_imgs):
    result_img = decoded_img * 255.0  # Denormalize to [0, 255]
    result_img = result_img.astype(np.uint8)
    result_path = os.path.join(result_folder, f"decoded_{i}.png")
    tf.keras.preprocessing.image.save_img(result_path, result_img)

# 모델 저장
model_save_path = "autoencoder_modelnew.h5"
autoencoder.save(model_save_path)
print(f"모델이 {model_save_path}에 저장되었습니다.")
