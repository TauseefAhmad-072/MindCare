import cv2, sys, hashlib
sys.path.insert(0, '.')
from modules.image_emotion.detector import FaceDetector
from modules.image_emotion.model import classify_faces_batch
from modules.image_emotion.enhance import enhance_face_crop, enhance_frame
from modules.image_emotion.config import ENHANCE_ENABLED

print(f"ENHANCE_ENABLED = {ENHANCE_ENABLED}")

img = cv2.imread(r'.\uploads\happy_face.jpg')
b = FaceDetector().detect(img)[0]
crop_raw = img[b.y:b.y+b.h, b.x:b.x+b.w]
print(f"raw crop shape: {crop_raw.shape}  hash: {hashlib.md5(crop_raw.tobytes()).hexdigest()[:8]}")

enhanced_frame = enhance_frame(img)
crop_from_enhanced_frame = enhanced_frame[b.y:b.y+b.h, b.x:b.x+b.w]
print(f"from-enhanced-frame shape: {crop_from_enhanced_frame.shape}  hash: {hashlib.md5(crop_from_enhanced_frame.tobytes()).hexdigest()[:8]}")

crop_enhanced = enhance_face_crop(crop_raw)
print(f"crop-enhanced shape: {crop_enhanced.shape}  hash: {hashlib.md5(crop_enhanced.tobytes()).hexdigest()[:8]}")

print("\n--- classify raw crop ---")
r = classify_faces_batch([crop_raw])[0]
print(f"  {r['emotion']}  {r['confidence']:.3f}")

print("\n--- classify from-enhanced-frame crop ---")
r = classify_faces_batch([crop_from_enhanced_frame])[0]
print(f"  {r['emotion']}  {r['confidence']:.3f}")

print("\n--- classify crop-enhanced ---")
r = classify_faces_batch([crop_enhanced])[0]
print(f"  {r['emotion']}  {r['confidence']:.3f}")