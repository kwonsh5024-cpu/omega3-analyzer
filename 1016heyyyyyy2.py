import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# ----------------------------
# 한글 폰트 설정 (Streamlit 호환)
# ----------------------------
font_path = os.path.join(os.getcwd(), "NanumGothic.ttf")
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
else:
    font_prop = None
    st.write("NanumGothic.ttf 파일 없음")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# ----------------------------
# Streamlit 기본 설정
# ----------------------------
st.set_page_config(page_title="오메가-3 산패 판정 시스템", page_icon="💊", layout="centered")
st.title("💊 오메가-3 색 기반 산패 판정 시스템")

normal_lab = np.array([75.0, 5.0, 25.0])  # 기준 밝은 황금빛

# ----------------------------
# 알약 영역 추출 + 그림자 제거
# ----------------------------
def extract_capsule_area(image: Image.Image):
    img = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([10, 40, 70])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    v_channel = hsv[:, :, 2]
    shadow_mask = v_channel < 80
    mask[shadow_mask] = 0

    masked = cv2.bitwise_and(img, img, mask=mask)
    return Image.fromarray(masked), mask

# ----------------------------
# 평균 LAB 색상 계산
# ----------------------------
def mean_lab_in_mask(image: Image.Image, mask):
    img_arr = np.array(image.convert("RGB")) / 255.0
    lab_arr = color.rgb2lab(img_arr)
    mask_bool = mask > 0
    if np.sum(mask_bool) == 0:
        return None
    mean_lab = np.mean(lab_arr[mask_bool], axis=0)
    return mean_lab

# ----------------------------
# LAB 변화 시각화
# ----------------------------
def plot_lab_differences(L_diff, a_diff, b_diff):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    diffs = [L_diff, a_diff, b_diff]
    labels = ['밝기 (L*)', '붉은기 (a*)', '노란기 (b*)']
    colors = ['gold', 'tomato', 'skyblue']

    bars = ax.bar(range(len(diffs)), diffs, color=colors)

    # x축 레이블 적용
    ax.set_xticks(range(len(labels)))
    if font_prop:
        ax.set_xticklabels(labels, fontproperties=font_prop)
    else:
        ax.set_xticklabels(labels)

    # 바 위 숫자 표시
    for bar, val in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val > 0 else -1),
                f"{val:.1f}", ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontproperties=font_prop if font_prop else None)

    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(-5, color='orange', linestyle='--', linewidth=1, alpha=0.8, label='L* ≤ -5 : 어두워짐(주의)')
    ax.axhline(4, color='red', linestyle='--', linewidth=1, alpha=0.8, label='a* ≥ +4 : 붉어짐(주의)')
    ax.axhline(-3, color='skyblue', linestyle='--', linewidth=1, alpha=0.8, label='b* ≤ -3 : 노란기 감소(주의)')

    if font_prop:
        ax.set_title("색 변화 방향 (밝기·붉은기·노란기)", fontsize=12, pad=10, fontproperties=font_prop)
        ax.set_ylabel("변화량 (Δ)", fontsize=10, fontproperties=font_prop)
        ax.legend(fontsize=8, loc='upper right', prop=font_prop)
    else:
        ax.set_title("색 변화 방향 (밝기·붉은기·노란기)", fontsize=12, pad=10)
        ax.set_ylabel("변화량 (Δ)", fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#ffffff")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)


# ----------------------------
# 산패 판정 로직
# ----------------------------
def judge_oxidation(mean_lab, normal_lab):
    deltaE = np.linalg.norm(mean_lab - normal_lab)
    L_diff = mean_lab[0] - normal_lab[0]
    a_diff = mean_lab[1] - normal_lab[1]
    b_diff = mean_lab[2] - normal_lab[2]

    warning_signs = []
    if L_diff <= -5: warning_signs.append("밝기 감소")
    if a_diff >= 4: warning_signs.append("붉은기 증가")
    if b_diff <= -3: warning_signs.append("노란기 감소")

    if L_diff > 0:
        status = "🟢 정상"
        desc = "밝기가 정상보다 높습니다. 조명 영향으로 판단됩니다."
    elif deltaE < 10:
        status = "🟢 정상"
        desc = "색 변화가 거의 없습니다. 산패되지 않은 상태입니다."
    elif deltaE < 25 and L_diff >= -3:
        status = "🟡 주의"
        desc = "약간의 색 변화가 있습니다. 환경 영향일 수 있습니다."
    else:
        status = "🔴 위험"
        desc = "명확한 색 변화가 확인되었습니다. 산패 가능성이 높습니다."

    if len(warning_signs) > 0:
        desc += f"  (감지된 변화: {', '.join(warning_signs)})"

    return deltaE, L_diff, a_diff, b_diff, status, desc

# ----------------------------
# 알약 여부 체크 함수 (면적 + 컨투어 + 비율)
# ----------------------------
def is_capsule(mask, image):
    mask_area = np.sum(mask)
    h, w = mask.shape
    total_area = h * w
    mask_ratio = mask_area / total_area

    if mask_ratio < 0.01:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest_contour)
    aspect_ratio = cw / ch
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < 500 or aspect_ratio < 0.3 or aspect_ratio > 3:
        return False

    return True

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("📸 **오메가-3 캡슐 사진을 업로드하면 자동 분석됩니다.**")
multi_files = st.file_uploader("여러 장의 사진도 업로드 가능", 
                               type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if multi_files:
    for file in multi_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"업로드 이미지: {file.name}", use_container_width=True)

        capsule_img, mask = extract_capsule_area(image)

        if not is_capsule(mask, image):
            st.warning("⚠️ 업로드된 사진에서 알약이 감지되지 않았습니다. 알약 사진만 올려주세요.")
            continue

        st.image(capsule_img, caption="🎯 알약 영역 추출 결과", use_container_width=True)

        mean_lab = mean_lab_in_mask(image, mask)
        if mean_lab is not None:
            deltaE, L_diff, a_diff, b_diff, status, desc = judge_oxidation(mean_lab, normal_lab)
            st.subheader(f"결과 판정: {status}")
            st.write(desc)
            st.markdown(f"""
            **ΔE (색차)**: {deltaE:.2f}  
            **L_diff (밝기 변화)**: {L_diff:.2f}  
            **a_diff (붉은기 변화)**: {a_diff:.2f}  
            **b_diff (노란기 변화)**: {b_diff:.2f}
            """)
            plot_lab_differences(L_diff, a_diff, b_diff)
            st.write("---")
        else:
            st.warning("⚠️ 알약 영역 인식 실패. 배경 단색 사진 사용 권장.")
else:
    st.info("오메가-3 캡슐 이미지를 업로드하면 결과가 표시됩니다.")






