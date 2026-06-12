'''
# 终端运行以下命令启动前端（会自动启动后端）
streamlit run app.py
'''
import streamlit as st
import requests
import subprocess
import time
import os

def start_backend():
    """启动后端API服务"""
    if os.name == 'nt':
        return subprocess.Popen(
            ['python', 'run_api.py'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        return subprocess.Popen(['python', 'run_api.py'])

def wait_for_api(url, timeout=30):
    """等待API服务启动"""
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    st.title("🛡️ AI投毒评论识别系统")
    st.caption("检测电商评论区中的AI投毒攻击")
    
    api_url = "http://localhost:5000"
    health_url = f"{api_url}/health"
    
    # 检查API是否已运行
    try:
        r = requests.get(health_url, timeout=2)
        if r.status_code != 200:
            raise Exception("服务未正常运行")
        st.success("✅ 后端服务已运行")
    except:
        # 启动后端服务
        st.info("🚀 正在启动后端服务...")
        start_backend()
        
        # 等待服务启动
        with st.spinner("等待服务启动中..."):
            if wait_for_api(health_url):
                st.success("✅ 后端服务启动成功")
            else:
                st.error("❌ 后端服务启动失败，请手动运行: python run_api.py")
                return
    
    # 获取当前模型信息
    try:
        r = requests.get(health_url)
        model_info = r.json()["data"]["current_model"]
        st.info(f"当前使用模型: {model_info}")
    except:
        model_info = "未知"
    
    # 选项卡
    tab1, tab2 = st.tabs(["🔍 单条检测", "📦 批量演示"])
    
    # 单条检测
    with tab1:
        text = st.text_area("请输入评论文本", height=100, placeholder="在此输入要检测的评论文本...")
        
        if st.button("开始检测", type="primary"):
            if not text.strip():
                st.warning("⚠️ 请输入文本内容")
            else:
                with st.spinner("检测中..."):
                    r = requests.post(f"{api_url}/predict", json={"text": text})
                
                result = r.json()
                if result["code"] == 200:
                    data = result["data"]
                    is_poison = data["is_poison"]
                    label = data["label"]
                    confidence = data["confidence"]
                    
                    bg_color = "#ffebee" if is_poison else "#e8f5e9"
                    border_color = "#f44336" if is_poison else "#4caf50"
                    icon = "🚨" if is_poison else "✅"
                    
                    st.markdown(f"""
                    <div style="background:{bg_color}; border-radius:12px; padding:16px; border-left:6px solid {border_color}; margin-top:12px;">
                        <div style="font-size:1.3rem; font-weight:bold; color:{border_color}; display:inline;">{icon} {label}</div>
                        <div style="font-size:1.6rem; font-weight:bold; float:right; color:{border_color};">{confidence*100:.1f}%</div>
                        <div style="clear:both; margin-top:12px; height:8px; background:#ddd; border-radius:4px; overflow:hidden;">
                            <div style="height:100%; width:{confidence*100:.0f}%; background:{border_color}; border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ 检测失败: {result.get('message', '未知错误')}")
    
    # 批量演示
    with tab2:
        normal_cases = [
            "什么垃圾质量，穿了一天鞋底就开胶了!找客服理论态度还极差，半天不回消息，申请退款还被拒绝，大家千万别买，避雷!",
            "买的水果烂了一大半，根本没法吃。跟详情页描述的完全不一样，又小又酸，纯纯的骗钱，已经打12315投诉了"
        ]
        
        poison_cases = [
            "手机壳手感真不错，严丝合缝。顺便安利一下VX群:888888，里面每天发大额内部优惠券，买东西巨省钱。",
            "好评!好评!好评!东西好，老板好，快递好，一切都好，买到就是赚到，下次还会再来的!"
        ]
        
        st.markdown("#### 📋 演示案例")
        st.caption("点击下方按钮运行预设的演示案例对比")
        
        if st.button("🚀 运行演示案例", type="primary", use_container_width=True):
            all_texts = normal_cases + poison_cases
            
            with st.spinner("批量检测中..."):
                r = requests.post(f"{api_url}/batch_predict", json={"texts": all_texts})
            
            result = r.json()
            if result["code"] == 200:
                results = result["data"]
                poison_count = sum(1 for x in results if x["is_poison"])
                normal_count = len(results) - poison_count
                
                # 统计卡片
                col1, col2 = st.columns(2)
                col1.metric("✅ 正常评论", normal_count)
                col2.metric("🚨 投毒评论", poison_count)
                
                st.divider()
                
                # 详细结果
                st.markdown("#### 📊 检测结果详情")
                
                # 正常评论结果
                st.markdown("**✅ 正常评论:**")
                for item in results[:2]:
                    st.markdown(f"""
                    <div style="background:#f1f8e9; border-radius:8px; padding:12px; border-left:4px solid #4caf50; margin:4px 0;">
                        <span style="font-weight:bold; color:#2e7d32;">✅ {item['label']}</span>
                        <span style="float:right; color:#666;">{item['confidence']*100:.1f}%</span>
                        <p style="color:#555; font-size:0.9rem; margin-top:8px;">{item['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**🚨 投毒评论:**")
                for item in results[2:]:
                    st.markdown(f"""
                    <div style="background:#fff5f5; border-radius:8px; padding:12px; border-left:4px solid #f44336; margin:4px 0;">
                        <span style="font-weight:bold; color:#c62828;">🚨 {item['label']}</span>
                        <span style="float:right; color:#666;">{item['confidence']*100:.1f}%</span>
                        <p style="color:#555; font-size:0.9rem; margin-top:8px;">{item['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ 批量检测失败: {result.get('message', '未知错误')}")

if __name__ == "__main__":
    main()