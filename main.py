import cv2
import mediapipe as mp
import math
import numpy as np

# ============================================================================
# [1. 설정 및 튜닝 영역] - 이곳의 값을 변경하여 느낌을 조절하세요.
# ============================================================================

# 파일 경로
BG_IMAGE_PATH = 'background_asset/background_sea.jpg'

# 출력 화면 설정
WINDOW_W = 1280  # 디지털 창문 화면 너비
WINDOW_H = 720  # 디지털 창문 화면 높이

# 민감도 (Sensitivity)
XY_SENSITIVITY = 1.5  # 좌우/상하 움직임 배율 (클수록 배경이 많이 움직임)
Z_SENSITIVITY = 0.5  # 줌(Zoom) 민감도 (클수록 확/축소가 급격함)

# 부드러움 설정 (Smoothing / Jitter Reduction) - 0.0 ~ 1.0
# 값이 작을수록 부드럽지만 반응이 느려짐 (0.1 ~ 0.3 권장)
XY_SMOOTHING = 0.15
ZOOM_SMOOTHING = 0.02  # 줌은 더 부드럽게 처리하는 것이 멀미 방지에 좋음

# 데드존 (Dead Zone)
DEAD_ZONE_PIXEL = 5  # 중앙에서 이 픽셀만큼의 미세한 떨림은 무시함

# PiP (Picture-in-Picture) 웹캠 설정
PIP_SCALE = 0.25  # 메인 화면 대비 웹캠 크기 비율
PIP_MARGIN = 20  # 왼쪽 하단 여백

# ============================================================================

# 2. 초기화 및 준비
# ============================================================================

# 배경 이미지 로드
bg_origin = cv2.imread(BG_IMAGE_PATH)
if bg_origin is None:
    print(f"[오류] '{BG_IMAGE_PATH}' 파일을 찾을 수 없습니다. 실행 폴더에 이미지를 넣어주세요.")
    exit()

# 배경 이미지를 충분히 크게 리사이징 (줌인/아웃 및 이동을 커버하기 위해 화면의 3배로 설정)
bg_h_target, bg_w_target = int(WINDOW_H * 3), int(WINDOW_W * 3)
bg_origin = cv2.resize(bg_origin, (bg_w_target, bg_h_target))
bg_h, bg_w, _ = bg_origin.shape
bg_center_x, bg_center_y = bg_w // 2, bg_h // 2

# MediaPipe 설정
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# 전역 변수 (스무딩을 위한 이전 프레임 데이터 저장용)
prev_cx, prev_cy = 0, 0  # 이전 프레임의 얼굴 중심 좌표
prev_zoom = 1.0  # 이전 프레임의 줌 배율
base_eye_dist = 0  # 기준 거리 (초기값 0)

print("=== Digital Window Started ===")
print("Tip: 화면이 너무 가깝거나 멀게 느껴지면 'R' 키를 눌러 기준 거리를 재설정하세요.")

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        cam_h, cam_w, _ = image.shape

        # 3. 얼굴 추적 및 데이터 추출
        # ========================================================================
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True

        # 기본값 설정 (얼굴 미감지 시 중앙 유지)
        target_cx, target_cy = cam_w // 2, cam_h // 2
        target_zoom_raw = 1.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 추출 (385: 왼쪽 눈, 159: 오른쪽 눈)
                left_eye = face_landmarks.landmark[385]
                right_eye = face_landmarks.landmark[159]

                lx, ly = int(left_eye.x * cam_w), int(left_eye.y * cam_h)
                rx, ry = int(right_eye.x * cam_w), int(right_eye.y * cam_h)

                # [Raw Data] 현재 프레임의 날것 좌표 및 거리
                raw_cx = (lx + rx) // 2
                raw_cy = (ly + ry) // 2
                curr_eye_dist = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

                # 기준 거리 설정 (최초 1회 또는 리셋 시)
                if base_eye_dist == 0:
                    base_eye_dist = curr_eye_dist

                # 줌 비율 계산 (현재 거리 / 기준 거리)
                # 거리가 가까워지면(값이 커지면) > 1.0 (확대)
                ratio = curr_eye_dist / base_eye_dist
                target_zoom_raw = 1 + (ratio - 1) * Z_SENSITIVITY

                # 타겟 좌표 업데이트
                target_cx, target_cy = raw_cx, raw_cy

                # 시각화 (웹캠 화면에 눈과 중심점 그리기)
                cv2.line(image, (lx, ly), (rx, ry), (255, 0, 0), 1)
                cv2.circle(image, (raw_cx, raw_cy), 5, (0, 0, 255), -1)

        # 4. 데이터 후처리 (스무딩 & 데드존)
        # ========================================================================

        # (1) 초기값 세팅
        if prev_cx == 0: prev_cx, prev_cy = target_cx, target_cy

        # (2) 지수 이동 평균(EMA) 필터 적용 - 좌표
        # 공식: 현재값 = 목표값 * a + 이전값 * (1-a)
        smooth_cx = int(target_cx * XY_SMOOTHING + prev_cx * (1 - XY_SMOOTHING))
        smooth_cy = int(target_cy * XY_SMOOTHING + prev_cy * (1 - XY_SMOOTHING))

        # (3) 지수 이동 평균(EMA) 필터 적용 - 줌
        target_zoom_raw = max(0.5, min(target_zoom_raw, 2.5))  # 줌 제한 (0.5배~2.5배)
        smooth_zoom = target_zoom_raw * ZOOM_SMOOTHING + prev_zoom * (1 - ZOOM_SMOOTHING)

        # 다음 프레임을 위해 저장
        prev_cx, prev_cy = smooth_cx, smooth_cy
        prev_zoom = smooth_zoom

        # (4) 데드존 (Dead Zone) 적용
        # 화면 중앙으로부터의 거리를 구함
        diff_x = smooth_cx - (cam_w // 2)
        diff_y = smooth_cy - (cam_h // 2)

        if abs(diff_x) < DEAD_ZONE_PIXEL: diff_x = 0
        if abs(diff_y) < DEAD_ZONE_PIXEL: diff_y = 0

        # 5. 배경 이미지 처리 (Digital Window Rendering)
        # ========================================================================

        # (1) 이동량(Shift) 계산 - 정규화 후 출력 화면 크기에 맞춰 매핑
        norm_x = diff_x / cam_w
        norm_y = diff_y / cam_h

        shift_x = int(norm_x * WINDOW_W * XY_SENSITIVITY * -1)  # 반대 방향 이동
        shift_y = int(norm_y * WINDOW_H * XY_SENSITIVITY * -1)

        # (2) 크롭(Crop) 영역 계산
        # 줌 인(>1.0)이면 배경을 작게 잘라내야 함
        crop_w = int(WINDOW_W / smooth_zoom)
        crop_h = int(WINDOW_H / smooth_zoom)

        # 크롭 시작 좌표 (Top-Left)
        # 배경 중심 + 이동량 - (잘라낼 크기의 절반)
        tl_x = bg_center_x + shift_x - (crop_w // 2)
        tl_y = bg_center_y + shift_y - (crop_h // 2)

        # 좌표가 이미지 밖으로 나가지 않도록 제한 (Clamping)
        tl_x = max(0, min(tl_x, bg_w - crop_w))
        tl_y = max(0, min(tl_y, bg_h - crop_h))

        # (3) 이미지 자르기 및 리사이징 (최종 화면 생성)
        cropped_bg = bg_origin[tl_y:tl_y + crop_h, tl_x:tl_x + crop_w]
        final_view = cv2.resize(cropped_bg, (WINDOW_W, WINDOW_H), interpolation=cv2.INTER_LINEAR)

        # 6. PiP (Picture-in-Picture) 오버레이
        # ========================================================================
        pip_w = int(WINDOW_W * PIP_SCALE)
        pip_h = int(pip_w * (cam_h / cam_w))  # 비율 유지

        small_frame = cv2.resize(image, (pip_w, pip_h))

        # 테두리 및 텍스트
        cv2.rectangle(small_frame, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)
        status_text = "Tracking On" if face_detected else "No Face"
        cv2.putText(small_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 왼쪽 하단 위치 계산
        y1 = WINDOW_H - PIP_MARGIN - pip_h
        y2 = WINDOW_H - PIP_MARGIN
        x1 = PIP_MARGIN
        x2 = PIP_MARGIN + pip_w

        # 덮어쓰기
        final_view[y1:y2, x1:x2] = small_frame

        # 정보 표시
        info_txt = f"Zoom: x{smooth_zoom:.2f} | Shift: X{shift_x}, Y{shift_y}"
        cv2.putText(final_view, info_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 화면 출력
        cv2.imshow('Digital Window - Final Prototype', final_view)

        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC 종료
            break
        elif key == ord('r'):  # 'r' 리셋
            base_eye_dist = 0
            print("[Info] 기준 거리(Zero Point)가 재설정되었습니다.")

cap.release()
cv2.destroyAllWindows()