import streamlit as st

class PremiumLoader:
    """
    A premium, reusable SaaS-style loading component replacing native st.spinner.
    Uses context manager to elegantly show/hide, and allows real-time progress updates.
    """
    def __init__(self, title="Building Your Financial Twin", initial_subtitle="Initializing...", start_progress=0):
        self.title = title
        self.subtitle = initial_subtitle
        self.progress = start_progress
        self.container = st.empty()
        
    def __enter__(self):
        self._render()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.empty()
        
    def update(self, progress: int, subtitle: str):
        """Update the progress bar and the subtitle dynamically."""
        self.progress = min(max(progress, 0), 100) # clamp between 0-100
        self.subtitle = subtitle
        self._render()
        
    def _render(self):
        html = f"""
        <style>
        .premium-loader-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 3rem 1rem;
            animation: fadeIn 0.5s ease-out forwards;
        }}
        .premium-loader-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 2.5rem;
            width: 100%;
            max-width: 450px;
            text-align: center;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            position: relative;
            overflow: hidden;
        }}
        .premium-loader-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #4F8CFF, #00D4FF, #22C55E);
            background-size: 200% 200%;
            animation: shimmerGradient 3s infinite linear;
        }}
        .ai-pulse-icon {{
            width: 72px;
            height: 72px;
            background: linear-gradient(135deg, rgba(79,140,255,0.15), rgba(0,212,255,0.15));
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto 1.5rem auto;
            border: 1px solid rgba(79, 140, 255, 0.5);
            box-shadow: 0 0 20px rgba(79, 140, 255, 0.3);
            animation: pulseGlow 2s infinite ease-in-out;
            font-size: 32px;
        }}
        .loader-title {{
            color: #F8FAFC;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            letter-spacing: -0.015em;
        }}
        .loader-subtitle {{
            color: #94A3B8;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            font-weight: 400;
            margin-bottom: 2rem;
            min-height: 1.5rem;
            transition: opacity 0.3s ease;
        }}
        .progress-track {{
            background: rgba(255,255,255,0.04);
            border-radius: 99px;
            height: 6px;
            width: 100%;
            overflow: hidden;
            margin-bottom: 0.75rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4F8CFF, #00D4FF);
            border-radius: 99px;
            width: {self.progress}%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 0 12px rgba(79, 140, 255, 0.5);
        }}
        .progress-fill::after {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
            transform: translateX(-100%);
            animation: shimmerFill 1.5s infinite;
        }}
        .progress-text {{
            color: #4F8CFF;
            font-size: 0.85rem;
            font-weight: 700;
            text-align: right;
            font-family: monospace;
        }}
        
        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(79, 140, 255, 0.4); transform: scale(0.98); }}
            50% {{ box-shadow: 0 0 25px 10px rgba(79, 140, 255, 0); transform: scale(1.02); }}
            100% {{ box-shadow: 0 0 0 0 rgba(79, 140, 255, 0); transform: scale(0.98); }}
        }}
        @keyframes shimmerFill {{
            100% {{ transform: translateX(100%); }}
        }}
        @keyframes shimmerGradient {{
            0% {{ background-position: 200% center; }}
            100% {{ background-position: -200% center; }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px); filter: blur(5px); }}
            to {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
        }}
        </style>
        
        <div class="premium-loader-wrapper">
            <div class="premium-loader-card">
                <div class="ai-pulse-icon"><i class="fa-solid fa-chart-line" style="color:#00D4FF;"></i></div>
                <div class="loader-title">{self.title}</div>
                <div class="loader-subtitle">{self.subtitle}</div>
                
                <div class="progress-track">
                    <div class="progress-fill"></div>
                </div>
                <div class="progress-text">{self.progress}%</div>
            </div>
        </div>
        """
        self.container.markdown(html, unsafe_allow_html=True)


def render_skeleton(type="chart", height="400px"):
    """
    Renders elegant shimmering skeleton placeholders for UI components.
    Available types: 'chart', 'kpi', 'box'.
    """
    style = f"""
    <style>
    .skeleton-box {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
        width: 100%;
        margin-bottom: 1.5rem;
    }}
    .skeleton-box::after {{
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        transform: translateX(-100%);
        background: linear-gradient(
            90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.04) 50%, 
            transparent 100%
        );
        animation: skelShimmer 2s infinite cubic-bezier(0.4, 0, 0.2, 1);
    }}
    @keyframes skelShimmer {{
        100% {{ transform: translateX(100%); }}
    }}
    
    .skel-header {{ height: 20px; width: 35%; background: rgba(255,255,255,0.04); border-radius: 4px; margin: 1.5rem; }}
    .skel-body-chart {{ height: {height}; margin: 0 1.5rem 1.5rem 1.5rem; background: rgba(255,255,255,0.02); border-radius: 8px; }}
    
    .skel-kpi-val {{ height: 40px; width: 60%; background: rgba(255,255,255,0.05); border-radius: 6px; margin: 0 1.5rem 0.75rem 1.5rem; }}
    .skel-kpi-sub {{ height: 16px; width: 35%; background: rgba(255,255,255,0.03); border-radius: 4px; margin: 0 1.5rem 1.5rem 1.5rem; }}
    </style>
    """
    
    if type == "chart":
        html = f"""
        <div class="skeleton-box">
            <div class="skel-header"></div>
            <div class="skel-body-chart"></div>
        </div>
        """
    elif type == "kpi":
        html = f"""
        <div class="skeleton-box">
            <div class="skel-header" style="height:14px; width:45%; margin-bottom: 1.25rem;"></div>
            <div class="skel-kpi-val"></div>
            <div class="skel-kpi-sub"></div>
        </div>
        """
    else:
        html = f"""
        <div class="skeleton-box" style="height: {height};"></div>
        """
        
    return st.markdown(style + html, unsafe_allow_html=True)




