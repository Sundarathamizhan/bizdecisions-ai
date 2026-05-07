import asyncio
import os
import plotly.graph_objects as go
from playwright.async_api import async_playwright

output_dir = r"c:\new"

html_template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
            margin: 0; padding: 20px;
        }}
        .card {{
            background-color: #1e293b;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #334155;
            width: fit-content;
            margin-bottom: 20px;
        }}
        .title {{
            color: #38bdf8;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-value {{
            color: {metric_color};
            font-size: 3rem;
            font-weight: 700;
            margin: 10px 0;
            text-align: center;
        }}
        .metric-label {{
            color: #94a3b8;
            font-size: 1rem;
            text-align: center;
        }}
        .feed-item {{
            display: flex;
            align-items: flex-start;
            gap: 15px;
            padding: 15px;
            border-bottom: 1px solid #334155;
        }}
        .feed-item:last-child {{
            border-bottom: none;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge.negative {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        .badge.positive {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}
        .badge.neutral {{ background: rgba(234, 179, 8, 0.2); color: #fbbf24; }}
        .feed-time {{ color: #94a3b8; font-size: 0.9rem; min-width: 70px; }}
        .feed-text {{ flex-grow: 1; }}
        .alert-bar {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            padding: 10px;
            border-radius: 4px;
            border-left: 4px solid #ef4444;
            margin-bottom: 15px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

async def generate_html_snapshots():
    # 1. Health Score Badge
    health_score_html = html_template.format(
        metric_color="#22c55e",
        content="""
        <div class="card" id="health-card" style="width: 300px;">
            <div class="metric-label">Business Health Score</div>
            <div class="metric-value">85 / 100</div>
            <div class="metric-label">Status: Healthy 🟢</div>
        </div>
        """
    )
    
    # 2. Social Monitor Feed
    social_monitor_html = html_template.format(
        metric_color="#38bdf8",
        content="""
        <div class="card" id="social-card" style="width: 800px;">
            <div class="title">📡 Live Social Media Sentinel</div>
            <div class="alert-bar">⚠️ ALERT (Product): 4 Negative mentions in the last 15 minutes!</div>
            
            <div class="feed-item">
                <div class="feed-time">10:42 AM</div>
                <div class="feed-text">"The new coffee blend is absolutely fantastic!"</div>
                <div><span class="badge positive">Product: Positive</span></div>
            </div>
            <div class="feed-item">
                <div class="feed-time">10:38 AM</div>
                <div class="feed-text">"Wait times are getting ridiculous at the downtown branch."</div>
                <div><span class="badge negative">Service: Negative</span></div>
            </div>
            <div class="feed-item">
                <div class="feed-time">10:31 AM</div>
                <div class="feed-text">"Wish the app loaded faster on older phones."</div>
                <div><span class="badge negative">Product: Negative</span></div>
            </div>
            <div class="feed-item">
                <div class="feed-time">10:15 AM</div>
                <div class="feed-text">"Prices are okay, nothing special but fair."</div>
                <div><span class="badge neutral">Price: Neutral</span></div>
            </div>
        </div>
        """
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Health Score
        await page.set_content(health_score_html)
        await page.locator("#health-card").screenshot(path=os.path.join(output_dir, "health_score.png"))
        
        # Social Monitor
        await page.set_content(social_monitor_html)
        await page.locator("#social-card").screenshot(path=os.path.join(output_dir, "social_monitor.png"))
        
        await browser.close()
        print("HTML Mocks generated successfully.")

def generate_radar_chart():
    categories = ['Product', 'Service', 'Price', 'Ambience']
    # Add first element to end to close the polygon
    r1 = [85, 75, 60, 90, 85]
    r2 = [60, 50, 70, 40, 60]
    cat_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r1, theta=cat_closed,
        fill='toself', name='BizDecisions AI',
        marker=dict(color='#38bdf8'),
        line=dict(color='#38bdf8')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=r2, theta=cat_closed,
        fill='toself', name='Competitor A',
        marker=dict(color='#ef4444'),
        line=dict(color='#ef4444')
    ))
    
    fig.update_layout(
        title="Competitor Comparison Across Dimensions",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#334155'),
            angularaxis=dict(gridcolor='#334155')
        ),
        paper_bgcolor='#1e293b',
        font=dict(color='#f8fafc', size=13),
        plot_bgcolor='#1e293b',
        showlegend=True,
        width=600, height=500
    )
    
    fig.write_image(os.path.join(output_dir, "radar_chart.png"), scale=2)
    print("Radar chart generated successfully.")

if __name__ == "__main__":
    generate_radar_chart()
    asyncio.run(generate_html_snapshots())
