import os
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def get_weekly_ranges(year=2024):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    ranges = []
    curr = start
    while curr <= end:
        week_end = min(curr + timedelta(days=6), end)
        ranges.append((curr.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d')))
        curr = week_end + timedelta(days=1)
    return ranges

def scrape_imdb_movies():
    """Scrape IMDb 2024 movies week-by-week to get all 21k+ results"""
    
    print("🚀 Starting Advanced IMDb Scraper for 2024 Movies...")
    
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    all_movies = []
    output_file = 'imdb_movies_2024.csv'
    
    # Load existing data if available to avoid duplicates and allow resuming
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        all_movies = existing_df.to_dict('records')
        print(f"📦 Loaded {len(all_movies)} existing movies from {output_file}")
    
    existing_titles = {m['Movie_Name'] for m in all_movies}
    
    weeks = get_weekly_ranges(2024)
    
    try:
        for start_date, end_date in weeks:
            print(f"\n📅 Period: {start_date} to {end_date}")
            
            # URL for feature films released in this week
            url = f"https://www.imdb.com/search/title/?title_type=feature&release_date={start_date},{end_date}&sort=num_votes,desc"
            
            driver.get(url)
            wait = WebDriverWait(driver, 15)
            
            # Click "See more" until all movies in this week are loaded
            consecutive_errors = 0
            while True:
                try:
                    # Scroll to bottom
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    
                    # Look for "See more" button
                    more_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button.ipc-see-more__button")))
                    
                    if "more" in more_button.text.lower():
                        driver.execute_script("arguments[0].scrollIntoView(true);", more_button)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", more_button)
                        print(f"   Clicked 'See more'... Total cards so far: {len(driver.find_elements(By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item'))}", end='\r')
                        time.sleep(random.uniform(2, 3))
                        consecutive_errors = 0
                    else:
                        break
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors > 2: # Try a few times in case of slow loading
                        break
                    time.sleep(2)

            # Extract data
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            movie_cards = soup.select("li.ipc-metadata-list-summary-item")
            
            new_count = 0
            for card in movie_cards:
                try:
                    title_tag = card.select_one("h3.ipc-title__text")
                    if not title_tag: continue
                    full_title = title_tag.get_text()
                    title = full_title.split('. ', 1)[-1] if '. ' in full_title else full_title
                    
                    if title in existing_titles:
                        continue
                        
                    storyline_tag = card.select_one(".ipc-html-content-inner-div")
                    storyline = storyline_tag.get_text().strip() if storyline_tag else "No storyline available"
                    
                    all_movies.append({
                        'Movie_Name': title,
                        'Storyline': storyline,
                        'Release_Period': f"{start_date} to {end_date}"
                    })
                    existing_titles.add(title)
                    new_count += 1
                except:
                    continue
            
            print(f"✨ Collected {new_count} new movies. Total: {len(all_movies)}")
            
            # Periodic save
            if new_count > 0:
                pd.DataFrame(all_movies).to_csv(output_file, index=False, encoding='utf-8')
                
    except Exception as e:
        print(f"\n🛑 Critical Error: {e}")
    finally:
        driver.quit()
        
    if all_movies:
        df = pd.DataFrame(all_movies)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n SUCCESS! Total movies in dataset: {len(df)}")
        return df
    return None

if __name__ == "__main__":
    scrape_imdb_movies()