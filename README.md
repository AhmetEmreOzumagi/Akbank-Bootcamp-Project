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

- ## 🌐 Linkler
- **Kaggle Notebook (Supervised):** [Akbank Bootcamp Notebook](https://www.kaggle.com/code/ahmetemreozumagi/akbank-bootcamp)  
- **GitHub Repo (Ahmet Emre Özumagil):** [Akbank Bootcamp Project](https://github.com/AhmetEmreOzumagi/Akbank-Bootcamp-Project)  
- **GitHub Repo (Elif Büşra Çaylan):** [Akbank Bootcamp Project](https://github.com/ebus99/Akbank-Bootcamp_Project)


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

## ⚙️ Nasıl Çalıştırılır?

Bu projeyi kendi bilgisayarınızda çalıştırmak için adımlar şunlardır:

1. Projeyi bilgisayarınıza indirin (repo’yu kopyalamak için):  
   git clone https://github.com/AhmetEmreOzumagi/Akbank-Bootcamp-Project.git

2. İndirilen klasörün içine ve ardından UI klasörüne gidin:  
   cd Akbank-Bootcamp-Project  
   cd UI

3. Gerekli kütüphaneleri yükleyin:  
   Eğer requirements.txt dosyası varsa:  
   pip install -r requirements.txt  
   Eğer yoksa temel kütüphaneleri yüklemek için:  
   pip install streamlit tensorflow pillow numpy

4. Streamlit arayüzünü başlatın:  
   streamlit run app.py

5. Komutu çalıştırdıktan sonra tarayıcıda bir web arayüzü açılacaktır.  
   Bu arayüz üzerinden bir resim yükleyerek modelin tahmin sonucunu görebilirsiniz.

