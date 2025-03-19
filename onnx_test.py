import onnxruntime
import numpy as np

# ONNX 모델 로드
onnx_model_path = "tinyyolov2-8.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 및 출력 정보 확인
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"✅ ONNX 모델 입력 노드: {input_name}")
print(f"✅ ONNX 모델 출력 노드: {output_name}")

# 더미 입력 데이터 생성 (YOLOv2 입력 크기: 1 x 3 x 416 x 416)
dummy_input = np.random.rand(1, 3, 416, 416).astype(np.float32)

# ONNX 추론 실행
output = session.run([output_name], {input_name: dummy_input})

print(f"✅ ONNX 모델 추론 성공! 출력 크기: {output[0].shape}")
