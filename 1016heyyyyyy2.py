import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import urllib.request, os, zipfile

# ----------------------------
# 0️⃣ 한글 폰트 설정 (Streamlit Cloud 대응)
# ----------------------------
font_path = "/tmp/NotoSansCJK-Regular.ttc"
if not os.path.exists(font_path):
    url = "https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip"
    zip_path = "/tmp/NotoSansCJK.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/tmp/")
    font_path = "/tmp/NotoSansCJK-Regular.ttc"

prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1️⃣ 기본 설정
# ----------------------------
st.set_page_config(page_title="오메가-3 산패 판정 시스템", page_icon="💊", layout="centered")
st.title("💊 오메가-3 색 기반 산패 판정 시스템 (v3.4 배포용)")

# 정상 기준값 (밝은 황금빛)
normal_lab = np.array([75.0, 5.0, 25.0])

# ----------------------------
# 2️⃣ 알약 영역 추출 + 그림자 제거
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
# 3️⃣ 평균 LAB 색상 계산
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
# 4️⃣ LAB 변화 시각화 그래프
# ----------------------------
def plot_lab_differences(L_diff, a_diff, b_diff):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    diffs = [L_diff, a_diff, b_diff]
    labels = ['밝기 (L*)', '붉은기 (a*)', '노란기 (b*)']
    colors = ['gold', 'tomato', 'skyblue']

    bars = ax.bar(labels, diffs, color=colors)
    for bar, val in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val > 0 else -1),
                f"{val:.1f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylim(-15, 15)
    ax.set_title("색 변화 방향 (밝기·붉은기·노란기)", fontsize=12, pad=10)
    ax.set_ylabel("변화량 (Δ)", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig)

# ----------------------------
# 5️⃣ 산패 판정 로직
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
# 6️⃣ Streamlit UI
# ----------------------------
st.markdown("📸 **오메가-3 캡슐 사진을 업로드하면 자동으로 분석이 시작됩니다.**")
multi_files = st.file_uploader("여러 장 업로드 가능", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if multi_files:
    for file in multi_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"업로드된 이미지: {file.name}", use_column_width=True)

        capsule_img, mask = extract_capsule_area(image)
        st.image(capsule_img, caption="🎯 알약 영역 추출 결과", use_column_width=True)

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
            st.warning("⚠️ 알약 영역을 인식하지 못했습니다. 배경이 단색인 사진을 사용해주세요.")
else:
    st.info("오메가-3 캡슐 이미지를 업로드하면 결과가 표시됩니다.")



