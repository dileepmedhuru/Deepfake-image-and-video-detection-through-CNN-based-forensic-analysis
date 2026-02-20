from tensorflow.keras.models import load_model
import numpy as np
import cv2

MODEL_PATH = r'C:\Users\medhu\Desktop\project\Forgery_detection\ml_models\cnn_model.h5'

model = load_model(MODEL_PATH)

print("=== MODEL ARCHITECTURE ===")
model.summary()

print("\n=== TESTING DIFFERENT PREPROCESSINGS ===")

# Load a real image from uploads folder to test
import os, glob
# Try to find any image in uploads
upload_dirs = [
    r'C:\Users\medhu\Desktop\project\Forgery_detection\backend\uploads\images',
    r'C:\Users\medhu\Desktop\project\Forgery_detection\backend\uploads',
]
test_img_path = None
for d in upload_dirs:
    if os.path.exists(d):
        imgs = glob.glob(os.path.join(d, '**', '*.jpg'), recursive=True) + \
               glob.glob(os.path.join(d, '**', '*.png'), recursive=True)
        if imgs:
            test_img_path = imgs[0]
            break

if test_img_path:
    print(f"\nUsing test image: {test_img_path}")
    img = cv2.imread(test_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Method 1: /255 normalization (current)
    m1 = img_resized.astype(np.float32) / 255.0
    m1 = np.expand_dims(m1, 0)
    p1 = float(model.predict(m1, verbose=0)[0][0])
    print(f"Method 1 (/255):              pred={p1:.6f}  conf={max(p1,1-p1)*100:.2f}%")

    # Method 2: /127.5 - 1  (MobileNet/Xception style)
    m2 = img_resized.astype(np.float32) / 127.5 - 1.0
    m2 = np.expand_dims(m2, 0)
    p2 = float(model.predict(m2, verbose=0)[0][0])
    print(f"Method 2 (/127.5 - 1):        pred={p2:.6f}  conf={max(p2,1-p2)*100:.2f}%")

    # Method 3: ImageNet mean subtraction
    m3 = img_resized.astype(np.float32)
    m3[:,:,0] -= 103.939
    m3[:,:,1] -= 116.779
    m3[:,:,2] -= 123.68
    m3 = np.expand_dims(m3, 0)
    p3 = float(model.predict(m3, verbose=0)[0][0])
    print(f"Method 3 (ImageNet mean sub): pred={p3:.6f}  conf={max(p3,1-p3)*100:.2f}%")

    # Method 4: keras preprocess_input for VGG/ResNet
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
    m4 = img_resized.astype(np.float32)
    m4 = vgg_pre(np.expand_dims(m4, 0))
    p4 = float(model.predict(m4, verbose=0)[0][0])
    print(f"Method 4 (VGG preprocess):    pred={p4:.6f}  conf={max(p4,1-p4)*100:.2f}%")

    # Method 5: no normalization at all
    m5 = img_resized.astype(np.float32)
    m5 = np.expand_dims(m5, 0)
    p5 = float(model.predict(m5, verbose=0)[0][0])
    print(f"Method 5 (no normalization):  pred={p5:.6f}  conf={max(p5,1-p5)*100:.2f}%")

    print("\nThe method with prediction FURTHEST from 0.5 is likely the correct one.")
    results = [('Method1 /255', p1), ('Method2 /127.5-1', p2),
               ('Method3 ImageNet', p3), ('Method4 VGG', p4), ('Method5 None', p5)]
    best = max(results, key=lambda x: abs(x[1] - 0.5))
    print(f"BEST METHOD: {best[0]}  (pred={best[1]:.6f}, furthest from 0.5)")

else:
    print("No test image found in uploads. Upload an image first and re-run.")

print("\n=== FIRST FEW LAYERS ===")
for i, layer in enumerate(model.layers[:8]):
    print(f"Layer {i}: {layer.name:30s}  type={type(layer).__name__}")
    if hasattr(layer, 'get_config'):
        cfg = layer.get_config()
        if 'mean' in str(cfg) or 'std' in str(cfg) or 'scale' in str(cfg):
            print(f"         Config hint: {cfg}")