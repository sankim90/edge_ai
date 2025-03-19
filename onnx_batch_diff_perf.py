import time
import onnxruntime
import numpy as np

# ONNX 모델 로드
onnx_model_path = "tinyyolov2-8.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 및 출력 정보 확인
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 배치 크기 설정
batch_sizes = [1, 2, 4, 8]  # 다양한 배치 크기 테스트

for batch_size in batch_sizes:
    # 입력 데이터 생성 (batch_size x 3 x 416 x 416)
    dummy_input = np.random.rand(batch_size, 3, 416, 416).astype(np.float32)

    # 추론 실행 및 FPS 측정
    start_time = time.time()
    session.run([output_name], {input_name: dummy_input})
    end_time = time.time()

    # 처리 시간 및 FPS 계산
    inference_time = end_time - start_time
    fps = batch_size / inference_time

    print(f"Batch Size: {batch_size} | Inference Time: {inference_time:.4f} sec | FPS: {fps:.2f}")
