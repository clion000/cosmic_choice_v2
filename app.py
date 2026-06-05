import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import t
from datetime import date

# ==========================================
# 1. 페이지 및 헬퍼 함수 설정
# ==========================================
st.set_page_config(page_title="COSMIC CHOICE (Fat Tail & Jump)", page_icon="⚡", layout="wide")

def format_price(value, currency):
    """표 데이터 찌꺼기를 제거하고 순수 실수(float) 2자리로 변환"""
    try:
        val = float(value.iloc[-1]) if isinstance(value, (pd.Series, pd.DataFrame)) else float(value)
    except:
        val = float(value)
    return f"{currency}{val:,.2f}"

st.title("⚡COSMIC CHOICE + 팻 테일 & 점프 확산 모델")
st.markdown("스튜던트 t-분포로 매핑된 양자 팻 테일(Fat Tail)과 돌발적인 폭등/폭락(Jump Diffusion) 시나리오를 통합한 심화 모델입니다.")

# ==========================================
# 2. 사이드바 - 파라미터 및 파일 멀티 선택
# ==========================================
st.sidebar.header("⚙️ 설정 및 데이터 입력")

ticker_input = st.sidebar.text_input("종목 코드 (Ticker)", value="AAPL")
TICKER = ticker_input.strip().upper()
currency_symbol = "₩" if TICKER.endswith(".KS") or TICKER.endswith(".KQ") else "$"

st.sidebar.markdown("---")

qrng_option = st.sidebar.selectbox(
    "🎲 사용할 양자 난수(QRNG) 선택",
    ["양자 난수 세트 1 (기본)", "양자 난수 세트 2 (추가)", "양자 난수 세트 3 (추가)", "PC에서 파일 직접 업로드"]
)

uploaded_file = None
if qrng_option == "PC에서 파일 직접 업로드":
    uploaded_file = st.sidebar.file_uploader("난수 데이터 파일 업로드 (.bin)", type=['bin'])

# [모델 하이퍼파라미터]
STEPS = 252              
NUM_PATHS = 1000         
DF = 4                    # 팻 테일(t-분포) 자유도

JUMP_PROB_DAILY = 5 / 252 # 1년에 약 5번의 돌발 점프 발생
JUMP_MEAN = -0.02         # 점프 시 평균 -2% 하락 충격
JUMP_VOL = 0.08           # 점프 크기의 변동성

# ==========================================
# 3. 메인 로직 연산
# ==========================================
if qrng_option == "PC에서 파일 직접 업로드" and uploaded_file is None:
    st.info("👈 좌측 사이드바에서 `.bin` 파일을 직접 업로드하면 시뮬레이션이 시작됩니다.")
else:
    with st.spinner("팻 테일 변환 및 점프 확산 연산 중..."):
        try:
            # --- A. 주가 데이터 로드 (버그 완벽 회피) ---
            ticker_obj = yf.Ticker(TICKER)
            data = ticker_obj.history(period="1y")
            
            if data.empty:
                st.error(f"❌ '{TICKER}' 종목의 데이터를 불러오지 못했습니다.")
                st.stop()
                
            close_prices = data['Close'].dropna()
            returns = np.log(close_prices / close_prices.shift(1)).dropna()

            if len(returns) < 5:
                st.error("과거 주가 데이터가 너무 적어 시뮬레이션이 불가능합니다.")
                st.stop()

            sigma_annual = float(np.std(returns)) * np.sqrt(252)
            mu_annual = (float(np.mean(returns)) * 252) + (0.5 * sigma_annual**2)

            mu_daily = mu_annual / 252
            sigma_daily = sigma_annual / np.sqrt(252)
            
            S0 = float(close_prices.iloc[-1])
            last_date = close_prices.index[-1]

            # --- B. QRNG 로드 및 [팻 테일] 매핑 ---
            if qrng_option == "양자 난수 세트 1 (기본)":
                target_file = "qrng_data_1.bin"
            elif qrng_option == "양자 난수 세트 2 (추가)":
                target_file = "qrng_data_2.bin"
            elif qrng_option == "양자 난수 세트 3 (추가)":
                target_file = "qrng_data_3.bin"
            else:
                target_file = None

            if target_file:
                try:
                    with open(target_file, "rb") as f:
                        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
                except FileNotFoundError:
                    st.error(f"서버에 '{target_file}' 파일이 없습니다. 깃허브에 해당 난수 파일을 업로드해주세요.")
                    st.stop()
            else:
                raw_data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)

            # 난수를 0~1 사이로 변환 후, 무한대 에러 방지를 위해 양끝을 살짝 자름
            u_data = np.clip(raw_data.astype(np.float32) / 255.0, 1e-7, 1 - 1e-7)

            required_z = STEPS * NUM_PATHS
            if len(u_data) < required_z:
                st.error(f"❌ 난수 데이터가 부족합니다! 필요: {required_z}개")
                st.stop()

            # [핵심 1] Box-Muller 대신 SciPy의 t.ppf를 사용하여 팻 테일(두꺼운 꼬리) 생성
            z_fat_raw = t.ppf(u_data[:required_z], DF)
            # 평소 변동성을 현실과 맞추기 위한 스케일링
            z_qrng_fat = z_fat_raw * np.sqrt((DF - 2) / DF)

            # --- C. [점프 확산]이 결합된 시뮬레이션 연산 ---
            results = np.zeros((STEPS, NUM_PATHS))
            results[0] = S0
            drift = mu_daily - 0.5 * sigma_daily**2

            for step in range(1, STEPS):
                # 1. 팻 테일 난수 추출
                z_t = z_qrng_fat[(step-1)*NUM_PATHS : step*NUM_PATHS]
                
                # 2. 돌발 점프(폭등/폭락) 이벤트 발생 여부 계산
                rand_jump = np.random.rand(NUM_PATHS)
                is_jump = rand_jump < JUMP_PROB_DAILY
                jump_sizes = np.random.normal(JUMP_MEAN, JUMP_VOL, NUM_PATHS)
                jump_multiplier = np.where(is_jump, np.exp(jump_sizes), 1.0)
                
                # 3. 팻 테일 GBM + 점프 충격 동시 적용
                results[step] = results[step-1] * np.exp(drift + sigma_daily * z_t) * jump_multiplier

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.stop()

        # ==========================================
        # 4. 결과 출력 및 시각화
        # ==========================================
        st.success(f"시뮬레이션 완료! (팻 테일 + 점프 확산 적용 / 소스: {qrng_option})")
        
        final_prices = results[-1, :]
        expected_avg = np.mean(final_prices)
        expected_median = np.median(final_prices)
        max_price = np.max(final_prices)
        min_price = np.min(final_prices)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=STEPS)
        final_date_str = future_dates[-1].strftime('%Y-%m-%d')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재 주가", format_price(S0, currency_symbol))
        col2.metric("중앙값 (Median)", format_price(expected_median, currency_symbol))
        # 팻 테일과 점프가 들어갔으므로 최고/최저 주가의 격차가 기존 모델보다 훨씬 커집니다.
        col3.metric("최고 예상 (Max - 대박 시나리오)", format_price(max_price, currency_symbol))
        col4.metric("최저 예상 (Min - 폭락 시나리오)", format_price(min_price, currency_symbol))

        tab1, tab2, tab3 = st.tabs(["📊 확률 원뿔 (Probability Cone)", "📈 경로 샘플 (Jump 관찰)", "📉 주가 분포 (Fat Tail 관찰)"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            # 꼬리가 길어졌으므로 관찰 범위를 1%~99%까지 확장
            quantiles = np.percentile(results, [1, 25, 50, 75, 99], axis=1)
            ax1.fill_between(future_dates, quantiles[0], quantiles[4], color='darkred', alpha=0.15, label='1% - 99% (Extreme Tail Risk)')
            ax1.fill_between(future_dates, quantiles[1], quantiles[3], color='royalblue', alpha=0.35, label='25% - 75% Range')
            ax1.plot(future_dates, quantiles[2], color='navy', linewidth=2, label='Median Path (50%)')
            ax1.set_title(f"[{TICKER}] Future Projection with Fat Tail & Jumps", fontsize=15, fontweight='bold')
            ax1.set_ylabel(f"Predicted Price ({currency_symbol})")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            st.pyplot(fig1)
            plt.close(fig1)

        with tab2:
            st.subheader("5 Random Paths (Notice the sudden Jumps)")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(future_dates, results[:, :5], alpha=0.8, linewidth=1.5) 
            ax2.set_title(f"[{TICKER}] 5 Random Paths (Look for vertical drops/spikes)", fontsize=15, fontweight='bold')
            ax2.set_ylabel(f"Predicted Price ({currency_symbol})")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            st.pyplot(fig2)
            plt.close(fig2)

        with tab3:
            st.subheader("Final Price Distribution (Notice the skewed tails)")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.hist(final_prices, bins=80, color='purple', edgecolor='black', alpha=0.7)
            ax3.axvline(S0, color='red', linestyle='dashed', linewidth=2, label=f"Current: {format_price(S0, currency_symbol)}")
            ax3.axvline(expected_avg, color='green', linestyle='dashed', linewidth=2, label=f"Average: {format_price(expected_avg, currency_symbol)}")
            ax3.axvline(expected_median, color='navy', linestyle='dashed', linewidth=2, label=f"Median: {format_price(expected_median, currency_symbol)}")
            ax3.set_title(f"[{TICKER}] Fat Tail Price Distribution on {final_date_str}", fontsize=15, fontweight='bold')
            ax3.set_xlabel(f"Final Price ({currency_symbol})")
            ax3.set_ylabel("Frequency")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig3)
            plt.close(fig3)