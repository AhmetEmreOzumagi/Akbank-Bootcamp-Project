# 🌍 Akbank Bootcamp Project

## 👥 Takım Üyeleri
- Ahmet Emre Özumağı 
- Elif Büşra Çaylan  

## 📂 Proje Açıklaması
Bu proje, **Intel Image Classification** veri seti üzerinde, **MobileNetV2 transfer learning** ve **fine-tuning** yöntemleriyle gerçekleştirilmiştir.  
Amaç, verilen görüntüleri altı sınıftan birine doğru şekilde sınıflandırmaktır:  
**Buildings, Forest, Glacier, Mountain, Sea, Street**

## 📊 Kullanılan Yöntemler
- Veri ön işleme (normalizasyon, augmentation)  
- Transfer learning (MobileNetV2, ImageNet ağırlıkları)  
- Fine-tuning (son katmanların yeniden eğitilmesi)  
- Callback mekanizmaları (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)  
- Model değerlendirmesi (Confusion Matrix, ROC, Classification Report)  
- Grad-CAM ile görselleştirme  

## 🧪 Sonuçlar
- **Test Accuracy:** %91.37  
- ROC AUC ve F1 skorları hesaplanmıştır.  
- Model, UI üzerinden yüklenen görsellerde de başarıyla çalışmaktadır.  

## 📂 Repository Yapısı
│
├── supervised.ipynb # Intel Image Classification (Supervised Learning)
├── unsupervised.ipynb # Gözetimsiz öğrenme örneği (K-Means & PCA)
├── UI/
│ ├── app.py # Streamlit arayüzü
│ └── best_model.h5 # Eğitilmiş model
├── README.md
├── LICENSE
└── .gitignore
