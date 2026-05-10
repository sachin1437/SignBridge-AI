import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load CSV
df = pd.read_csv(OUTPUT_CSV)
print(f'Total samples: {len(df)}')
print('\nSamples per class:')
print(df['label'].value_counts())

X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
le      = LabelEncoder()
y_enc   = le.fit_transform(y)
classes = le.classes_
print(f'\nClasses: {list(classes)}')

# Save classes
np.save('/kaggle/working/classes.npy', classes)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
print(f'\nTrain: {X_train.shape} | Test: {X_test.shape}')

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('/kaggle/working/model.h5', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Accuracy: {acc * 100:.2f}%')

y_pred = np.argmax(model.predict(X_test), axis=1)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=classes))

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('/kaggle/working/training_history.png')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes,
            yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()

print('\nAll files saved to /kaggle/working/')
print('Download: model.h5, classes.npy')