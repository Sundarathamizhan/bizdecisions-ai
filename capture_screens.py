import asyncio
from playwright.async_api import async_playwright
import time
import os

async def capture_streamlit():
    output_dir = r"c:\new"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 1920x1080 for desktop layout
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        
        print("Navigating to Streamlit app...")
        # Streamlit defaults to 8501
        await page.goto("http://localhost:8501", timeout=60000)
        
        # Wait for Streamlit to load
        # Sometimes Streamlit shows a "Please wait" or blank screen for a second
        await page.wait_for_timeout(5000)
        
        # 1. Capture Login Screen
        # wait for app layout to load
        await page.wait_for_timeout(5000)
        
        # 2. Capture Dashboard (Tab 1 by default)
        print("Capturing Dashboard Main Menu...")
        await page.screenshot(path=os.path.join(output_dir, "streamlit_dashboard.png"), full_page=True)
        
        # Streamlit tabs use data-baseweb='tab'
        tabs = await page.locator('[data-baseweb="tab"]').all()
        
        if len(tabs) >= 6:
            # 3. Tab 4 - Weekly AI Report (Health Score)
            print("Capturing Tab 4 (Health Score)...")
            await tabs[3].click()
            await page.wait_for_timeout(4000)
            await page.screenshot(path=os.path.join(output_dir, "health_score.png"), full_page=True)
            
            # 4. Tab 5 - Social Media Monitor
            print("Capturing Tab 5 (Social Monitor)...")
            await tabs[4].click()
            await page.wait_for_timeout(4000)
            await page.screenshot(path=os.path.join(output_dir, "social_monitor.png"), full_page=True)
            
            # 5. Tab 6 - Competitor Comparison (Radar Chart)
            print("Capturing Tab 6 (Radar Chart)...")
            await tabs[5].click()
            await page.wait_for_timeout(6000) # Give extra time for radar chart to render
            await page.screenshot(path=os.path.join(output_dir, "radar_chart.png"), full_page=True)
        else:
            print(f"Warning: Expected 6 tabs, found {len(tabs)}")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_streamlit())
