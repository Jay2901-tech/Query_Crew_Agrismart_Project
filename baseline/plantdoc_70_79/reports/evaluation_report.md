# AgriSmart-AI Evaluation Report

## Overall Metrics (Combined Field)
- **Macro Precision:** 0.9791
- **Macro Recall:** 0.9754
- **Macro F1-Score:** 0.9771
- **Balanced Accuracy:** 0.9754

## Domain-Specific Metrics

### plantvillage
- **Samples:** 5457
- **Macro F1-Score:** 0.9930
- **Balanced Accuracy:** 0.9922

### plantdoc
- **Samples:** 279
- **Macro F1-Score:** 0.7079
- **Balanced Accuracy:** 0.7337

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Apple___Apple_scab | 0.9857 | 0.9452 | 0.9650 | 73 |
| Apple___Black_rot | 1.0000 | 1.0000 | 1.0000 | 63 |
| Apple___Cedar_apple_rust | 0.9250 | 0.9737 | 0.9487 | 38 |
| Apple___healthy | 0.9943 | 1.0000 | 0.9972 | 175 |
| Blueberry___healthy | 0.9817 | 0.9877 | 0.9847 | 163 |
| Cherry___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 106 |
| Cherry___healthy | 0.9783 | 0.9677 | 0.9730 | 93 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9107 | 0.8500 | 0.8793 | 60 |
| Corn___Common_rust_ | 0.9776 | 0.9850 | 0.9813 | 133 |
| Corn___Northern_Leaf_Blight | 0.9262 | 0.9496 | 0.9378 | 119 |
| Corn___healthy | 0.9915 | 0.9915 | 0.9915 | 117 |
| Grape___Black_rot | 0.9764 | 0.9920 | 0.9841 | 125 |
| Grape___Esca_(Black_Measles) | 1.0000 | 0.9856 | 0.9928 | 139 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | 1.0000 | 109 |
| Grape___healthy | 1.0000 | 0.9804 | 0.9901 | 51 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | 1.0000 | 552 |
| Peach___Bacterial_spot | 1.0000 | 1.0000 | 1.0000 | 231 |
| Peach___healthy | 0.9583 | 0.9583 | 0.9583 | 48 |
| Pepper___bell___Bacterial_spot | 0.9907 | 0.9817 | 0.9862 | 109 |
| Pepper___bell___healthy | 0.9810 | 0.9936 | 0.9873 | 156 |
| Potato___Early_blight | 0.9561 | 0.9646 | 0.9604 | 113 |
| Potato___Late_blight | 0.9722 | 0.9545 | 0.9633 | 110 |
| Potato___healthy | 1.0000 | 0.9375 | 0.9677 | 16 |
| Raspberry___healthy | 1.0000 | 0.9804 | 0.9901 | 51 |
| Soybean___healthy | 0.9904 | 0.9961 | 0.9932 | 516 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 197 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | 1.0000 | 112 |
| Strawberry___healthy | 1.0000 | 1.0000 | 1.0000 | 58 |
| Tomato___Bacterial_spot | 0.9731 | 0.9644 | 0.9688 | 225 |
| Tomato___Early_blight | 0.9630 | 0.9455 | 0.9541 | 110 |
| Tomato___Late_blight | 0.9704 | 0.9704 | 0.9704 | 203 |
| Tomato___Leaf_Mold | 0.9455 | 0.9811 | 0.9630 | 106 |
| Tomato___Septoria_leaf_spot | 0.9545 | 0.9742 | 0.9643 | 194 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.9657 | 1.0000 | 0.9826 | 169 |
| Tomato___Target_Spot | 1.0000 | 0.9504 | 0.9745 | 141 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9945 | 0.9945 | 0.9945 | 545 |
| Tomato___Tomato_mosaic_virus | 0.9535 | 0.9318 | 0.9425 | 44 |
| Tomato___healthy | 0.9878 | 0.9759 | 0.9818 | 166 |