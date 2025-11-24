import cv2
import mediapipe as mp
import math

# 1. MediaPipe Face Mesh 설정
mp_face_mesh = mp.solutions.face_mesh
# 그리기 도구는 랜드마크 위치 확인용으로만 사용
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # ---------------------------------------------------------
                # 요청하신 랜드마크 추출 (385: 왼쪽 눈, 159: 오른쪽 눈)
                # ---------------------------------------------------------
                left_eye_point = face_landmarks.landmark[385]
                right_eye_point = face_landmarks.landmark[159]

                # 1. 픽셀 좌표로 변환
                lx, ly = int(left_eye_point.x * w), int(left_eye_point.y * h)
                rx, ry = int(right_eye_point.x * w), int(right_eye_point.y * h)

                # 2. 두 눈의 '중간 지점(Midpoint)' 계산 -> 이것이 새로운 cx, cy가 됩니다.
                cx = (lx + rx) // 2
                cy = (ly + ry) // 2

                # 3. 두 눈 사이의 거리 계산 (Z축 깊이 추정용)
                # 거리가 멀수록(값이 클수록) 카메라에 가깝다는 뜻입니다.
                eye_distance = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

                # ---------------------------------------------------------
                # 시각화 (디버깅용)
                # ---------------------------------------------------------

                # 양쪽 눈 포인트 표시 (파란색)
                cv2.circle(image, (lx, ly), 3, (255, 0, 0), -1)
                cv2.circle(image, (rx, ry), 3, (255, 0, 0), -1)

                # 두 눈을 잇는 선
                cv2.line(image, (lx, ly), (rx, ry), (255, 0, 0), 1)

                # 계산된 중심점 표시 (빨간색 큰 점) -> 디지털 창문의 기준점
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

                # 정보 출력
                info_text = f"Center:({cx}, {cy})  Depth:{int(eye_distance)}"
                cv2.putText(image, info_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Digital Window - Eye Tracking Demo', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()