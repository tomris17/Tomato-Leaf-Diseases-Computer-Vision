import os
import cv2
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tomato Leaf Diseases App", layout="centered")

st.title("Tomato Leaf Diseases Detection App")
st.write(
    "Bu uygulama, train/valid/test split yapısına sahip domates yaprağı hastalıkları veri setinin dizin yapısını kontrol eder ve örnek görselleri görselleştirir."
)

base_dir = "."
train_images_dir = os.path.join(base_dir, "train", "images")

if os.path.exists(train_images_dir):
    train_images = os.listdir(train_images_dir)
    st.success(f"Toplam eğitim görseli bulundu: {len(train_images)}")

    if len(train_images) > 0:
        st.subheader("Görsel Keşif Paneli")
        selected_img = st.selectbox("Bir Yaprak Görseli Seçin", train_images)
        
        if selected_img:
            sample_img_path = os.path.join(train_images_dir, selected_img)
            img = cv2.imread(sample_img_path)
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img_rgb)
                ax.axis("off")
                ax.set_title(f"Görsel: {selected_img}")
                st.pyplot(fig)
                
                st.write(f"Görsel Boyutu (Yükseklik, Genişlik, Kanal): {img.shape}")
            else:
                st.warning("Görsel okunamadı.")
else:
    st.warning("Eğitim görselleri dizini bulunamadı. Lütfen veri seti klasörlerinin çalışma alanında doğru yerleştirildiğinden emin olun.")