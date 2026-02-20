from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = r'C:\Users\medhu\Desktop\project\Forgery_detection\ml_models\cnn_model.h5'

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded OK")
print("Input shape: ", model.input_shape)
print("Output shape:", model.output_shape)

blank = np.zeros((1, 224, 224, 3), dtype=np.float32)
pred1 = float(model.predict(blank, verbose=0)[0][0])
print(f"Blank image  : {pred1:.6f}")

noise = np.random.rand(1, 224, 224, 3).astype(np.float32)
pred2 = float(model.predict(noise, verbose=0)[0][0])
print(f"Random noise : {pred2:.6f}")

white = np.ones((1, 224, 224, 3), dtype=np.float32)
pred3 = float(model.predict(white, verbose=0)[0][0])
print(f"White image  : {pred3:.6f}")

if pred1 == pred2 == pred3:
    print("\nDIAGNOSIS: Model outputs identical value for all inputs.")
    print("           The model is UNTRAINED or CORRUPTED.")
elif abs(pred1 - 0.5) < 0.01 and abs(pred2 - 0.5) < 0.01:
    print("\nDIAGNOSIS: Model stuck near 0.5 - likely undertrained.")
else:
    print("\nDIAGNOSIS: Model seems to be working (outputs vary).")
    print("           The issue may be in preprocessing.")