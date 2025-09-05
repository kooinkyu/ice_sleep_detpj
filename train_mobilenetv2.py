# mobilenet_train.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

# === 경로 설정 ===
data_dir = "/home/brian/Downloads/ice_mobilnet_data"
train_dir = os.path.join(data_dir, "Training", "cropped_real")
val_dir   = os.path.join(data_dir, "Validation", "cropped_real")

# === 파라미터 설정 ===
batch_size = 64
img_size   = (128, 128)
epochs     = 20
initial_lr = 1e-4

# === 데이터 제너레이터 ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === 클래스 가중치 ===
class_indices = train_generator.class_indices
class_names   = list(class_indices.keys())
class_counts  = [0] * len(class_names)
for label in train_generator.classes:
    class_counts[label] += 1

class_weight_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight = dict(enumerate(class_weight_arr))

# === 모델 구성 ===
base_model = MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=initial_lr)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# === 콜백 설정 ===
checkpoint_path = "mobilenet_best_model.keras"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# === 학습 ===
print("모델 학습 시작...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# === 평가 ===
print("모델 평가 중...")
val_loss, val_acc, val_topk = model.evaluate(val_generator, verbose=1)
print(f"검증 손실: {val_loss:.4f}")
print(f"검증 정확도: {val_acc:.4f}")
print(f"검증 Top-K 정확도: {val_topk:.4f}")

# === 예측 생성 ===
print("예측 생성 중...")
y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
class_names_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

# === 리포트 저장 ===
report = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names_ordered,
    digits=4
)
print("\n[RESULT] Classification report")
print(report)

cm = confusion_matrix(y_true, y_pred_classes)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

with open("classification_report.txt", "w") as f:
    f.write("[RESULT] Classification report\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(np.array2string(cm, max_line_width=200))
    f.write("\n\nNormalized confusion matrix (row-wise):\n")
    f.write(np.array2string(cm_norm, max_line_width=200, precision=4))

np.savetxt("confusion_matrix.csv", cm, fmt="%d", delimiter=",")
np.savetxt("confusion_matrix_normalized.csv", cm_norm, fmt="%.6f", delimiter=",")

# === 혼동 행렬 시각화 ===
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_ordered)
disp.plot(xticks_rotation='vertical', cmap='Blues', values_format='d')
plt.title("Confusion Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# === 학습 곡선 시각화 ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.ylabel('Accuracy'); plt.xlabel('Epoch')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.ylabel('Loss'); plt.xlabel('Epoch')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
plt.show()

# === ROC Curve ===
y_true_bin = label_binarize(y_true, classes=range(len(class_names_ordered)))
n_classes = y_true_bin.shape[1]

fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 8))
plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.4f})',
         color='navy', linestyle='--', linewidth=2)

colors = plt.cm.get_cmap('tab10', n_classes)
for i, color in zip(range(n_classes), colors.colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names_ordered[i]} (area = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-Class', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# === 모델 저장 ===
final_model_path = "mobilenet_final_model.keras"
model.save(final_model_path)
print(f"최종 모델이 {final_model_path}에 저장되었습니다.")
print("학습 완료!")
