#!/usr/bin/env python3
"""
Magyar receptek adatfeldolgozása és normalizálása - JAVÍTOTT VERZIÓ
DataFrame API kompatibilitási fix
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import re
import json

class HungarianRecipeProcessor:
    """Magyar receptek feldolgozása és normalizálása külső képekkel"""
    
    def __init__(self, csv_file_path="hungarian_recipes_github.csv"):
        self.csv_path = csv_file_path
        self.processed_data = None
        
    def load_and_validate_data(self):
        """CSV betöltése és validálása"""
        try:
            print(f"📊 Betöltés: {self.csv_path}")
            
            # Többféle encoding próbálása
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ Sikeres betöltés {encoding} encoding-gal")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print("❌ Nem sikerült betölteni egyik encoding-gal sem")
                return None
            
            print(f"✅ Sikeresen betöltve: {len(df)} recept")
            print(f"📋 Oszlopok: {list(df.columns)}")
            
            # Kötelező oszlopok ellenőrzése
            required_columns = ['name', 'ingredients', 'env_score', 'nutri_score', 'meal_score']
            optional_columns = ['instructions', 'images']
            
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                print(f"❌ Hiányzó kötelező oszlopok: {missing_required}")
                return None
            
            missing_optional = [col for col in optional_columns if col not in df.columns]
            if missing_optional:
                print(f"⚠️ Hiányzó opcionális oszlopok: {missing_optional}")
                # Létrehozzuk az üres oszlopokat
                for col in missing_optional:
                    df[col] = ''
            
            # Alapvető adatminőség ellenőrzés
            print(f"🔍 Adatminőség ellenőrzés:")
            print(f"   Üres nevek: {df['name'].isna().sum()}")
            print(f"   Üres összetevők: {df['ingredients'].isna().sum()}")
            print(f"   Env_score tartomány: {df['env_score'].min():.2f} - {df['env_score'].max():.2f}")
            print(f"   Nutri_score tartomány: {df['nutri_score'].min():.2f} - {df['nutri_score'].max():.2f}")
            print(f"   Meal_score tartomány: {df['meal_score'].min():.2f} - {df['meal_score'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Betöltési hiba: {e}")
            return None
    
    def normalize_environmental_scores(self, df):
        """Környezeti pontszámok normalizálása"""
        print("🌱 Környezeti pontszámok normalizálása...")
        
        # Környezeti score normalizálása (magasabb érték = rosszabb környezetileg)
        # Invertáljuk hogy magasabb = jobb legyen
        env_min = df['env_score'].min()
        env_max = df['env_score'].max()
        
        print(f"   Eredeti env_score tartomány: {env_min:.2f} - {env_max:.2f}")
        
        # Normalizálás 0-100 skálára (invertálva)
        df['ESI'] = 100 - ((df['env_score'] - env_min) / (env_max - env_min) * 100)
        
        print(f"   Normalizált env_score tartomány: {df['ESI'].min():.2f} - {df['ESI'].max():.2f}")
        
        return df
    
    def normalize_other_scores(self, df):
        """Egyéb pontszámok normalizálása"""
        print("📊 Egyéb pontszámok normalizálása...")
        
        # Nutri_score (már 0-100 skálán kellene lennie)
        nutri_min, nutri_max = df['nutri_score'].min(), df['nutri_score'].max()
        print(f"   nutri_score tartomány: {nutri_min:.2f} - {nutri_max:.2f}")
        
        if nutri_max <= 100:
            df['HSI'] = df['nutri_score']  # Health Score Index
            print("   nutri_score már normalizált")
        else:
            df['HSI'] = (df['nutri_score'] / nutri_max) * 100
            print("   nutri_score normalizálva")
        
        # Meal_score (népszerűség/ízletesség)
        meal_min, meal_max = df['meal_score'].min(), df['meal_score'].max()
        print(f"   meal_score tartomány: {meal_min:.2f} - {meal_max:.2f}")
        
        if meal_max <= 100:
            df['PPI'] = df['meal_score']  # Popularity/Preference Index
            print("   meal_score már normalizált")
        else:
            df['PPI'] = (df['meal_score'] / meal_max) * 100
            print("   meal_score normalizálva")
        
        return df
    
    def calculate_composite_score(self, df):
        """Kompozit pontszám számítása"""
        print("🔢 Kompozit pontszám számítása...")
        
        # Súlyozott átlag: Környezet 40%, Egészség 40%, Népszerűség 20%
        df['composite_score'] = (
            df['ESI'] * 0.4 +    # Environmental Score Index
            df['HSI'] * 0.4 +    # Health Score Index  
            df['PPI'] * 0.2      # Popularity/Preference Index
        )
        
        print(f"   Kompozit score tartomány: {df['composite_score'].min():.2f} - {df['composite_score'].max():.2f}")
        print(f"   Átlagos kompozit score: {df['composite_score'].mean():.2f}")
        
        return df
    
    def clean_text_data(self, df):
        """Szöveges adatok tisztítása - JAVÍTOTT VERZIÓ"""
        print("🧹 Adatok tisztítása...")
        
        try:
            # JAVÍTÁS: .str accessor helyett direct pandas műveletek
            
            # Üres értékek kezelése
            df['name'] = df['name'].fillna('Névtelen recept')
            df['ingredients'] = df['ingredients'].fillna('Ismeretlen összetevők')
            df['instructions'] = df['instructions'].fillna('Nincs útmutató')
            df['images'] = df['images'].fillna('')
            
            # Szöveges mezők tisztítása (biztonságos módszer)
            for col in ['name', 'ingredients', 'instructions']:
                if col in df.columns:
                    # Pandas Series.str helyett apply használata
                    df[col] = df[col].astype(str).apply(lambda x: x.strip() if isinstance(x, str) else str(x))
            
            # Recipe ID hozzáadása
            df['recipeid'] = range(1, len(df) + 1)
            
            # Oszlop átnevezés
            df = df.rename(columns={'name': 'title'})
            
            print(f"✅ Tisztítva: {len(df)} recept készenléti állapotban")
            
            return df
            
        except Exception as e:
            print(f"⚠️ Tisztítási hiba: {e}")
            # Fallback: alapvető tisztítás
            df['recipeid'] = range(1, len(df) + 1)
            df = df.rename(columns={'name': 'title'})
            return df
    
    def process_image_urls(self, df):
        """Kép URL-ek feldolgozása"""
        print("🖼️ Kép URL-ek feldolgozása...")
        
        def process_single_image_url(images_string):
            """Egy kép URL feldolgozása"""
            if pd.isna(images_string) or not images_string:
                return '/static/images/recipe_placeholder.jpg'
            
            # Ha string, akkor split by comma és első URL
            if isinstance(images_string, str):
                urls = images_string.split(',')
                first_url = urls[0].strip().strip('"').strip("'")
                
                # Ellenőrzés hogy valós URL-e
                if first_url.startswith('http'):
                    print(f"   🖼️ Kép URL: {first_url[:60]}...")
                    return first_url
            
            # Fallback
            return '/static/images/recipe_placeholder.jpg'
        
        # Biztonságos apply művelet
        try:
            df['images'] = df['images'].apply(process_single_image_url)
        except Exception as e:
            print(f"⚠️ Kép feldolgozási hiba: {e}")
            df['images'] = '/static/images/recipe_placeholder.jpg'
        
        return df
    
    def create_user_study_sample(self, df, sample_size=50):
        """User study minta létrehozása kiegyensúlyozott kompozit score-ral"""
        print(f"🎯 User study minta létrehozása ({sample_size} recept)...")
        
        # Kompozit score quartile-ok
        df['score_quartile'] = pd.qcut(df['composite_score'], 
                                     q=4, 
                                     labels=['low', 'medium', 'high', 'very_high'])
        
        # Kiegyensúlyozott mintavételezés
        sample_per_quartile = sample_size // 4
        remainder = sample_size % 4
        
        sampled_dfs = []
        for i, quartile in enumerate(['low', 'medium', 'high', 'very_high']):
            quartile_data = df[df['score_quartile'] == quartile]
            
            # Extra minta az első quartile-nek ha maradék van
            current_sample_size = sample_per_quartile + (1 if i < remainder else 0)
            
            if len(quartile_data) >= current_sample_size:
                sampled = quartile_data.sample(n=current_sample_size)
            else:
                sampled = quartile_data
            
            sampled_dfs.append(sampled)
            print(f"   {quartile}: {len(sampled)} recept")
        
        final_sample = pd.concat(sampled_dfs, ignore_index=True)
        print(f"✅ User study minta kész: {len(final_sample)} recept")
        
        return final_sample
    
    def generate_statistics_report(self, df):
        """Statisztikai riport generálása"""
        print(f"\n📊 ADATSTATISZTIKÁK")
        print("=" * 50)
        print(f"📈 Receptek száma: {len(df)}")
        print(f"📋 Oszlopok: {len(df.columns)}")
        
        # Score statisztikák
        for score_name, col_name in [('HSI', 'HSI'), ('ESI', 'ESI'), ('PPI', 'PPI'), ('composite_score', 'composite_score')]:
            if col_name in df.columns:
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                
                print(f"\n{score_name}:")
                print(f"   Átlag: {mean_val:.2f} ± {std_val:.2f}")
                print(f"   Tartomány: {min_val:.2f} - {max_val:.2f}")
        
        # Top receptek
        if 'composite_score' in df.columns and len(df) > 0:
            print(f"\n🏆 TOP 5 RECEPT (kompozit score):")
            top_recipes = df.nlargest(5, 'composite_score')[['title', 'composite_score']]
            for _, recipe in top_recipes.iterrows():
                title = recipe['title'][:40] + ('...' if len(recipe['title']) > 40 else '')
                print(f"   {title:<40} | Score: {recipe['composite_score']:.1f}")
        
        # Adatminőség
        print(f"\n🔍 ADATMINŐSÉG:")
        print(f"   Hiányzó címek: {df['title'].isna().sum()}")
        print(f"   Hiányzó összetevők: {df['ingredients'].isna().sum()}")
        if 'instructions' in df.columns:
            print(f"   Hiányzó instrukciók: {df['instructions'].isna().sum()}")
    
    def process_all(self, output_path="data/processed_recipes.csv", sample_size=50):
        """Teljes feldolgozási pipeline"""
        print("🚀 MAGYAR RECEPTEK FELDOLGOZÁSA")
        print("=" * 50)
        
        # 1. Betöltés és validálás
        df = self.load_and_validate_data()
        if df is None:
            return False
        
        # 2. Környezeti pontszámok normalizálása
        df = self.normalize_environmental_scores(df)
        
        # 3. Egyéb pontszámok normalizálása
        df = self.normalize_other_scores(df)
        
        # 4. Kompozit pontszám számítása
        df = self.calculate_composite_score(df)
        
        # 5. Szöveges adatok tisztítása (JAVÍTOTT)
        df = self.clean_text_data(df)
        
        # 6. Kép URL-ek feldolgozása
        df = self.process_image_urls(df)
        
        # 7. User study minta létrehozása
        self.processed_data = self.create_user_study_sample(df, sample_size)
        
        # 8. Statisztikák
        self.generate_statistics_report(self.processed_data)
        
        # 9. Mentés
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"\n💾 Feldolgozott adatok mentve: {output_path}")
            print(f"📁 Fájlméret: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            # 10. Mintaadatok kiírása
            print(f"\n📋 MINTA RECEPTEK:")
            for i in range(min(3, len(self.processed_data))):
                recipe = self.processed_data.iloc[i]
                print(f"   {i+1}. {recipe['title']}")
                print(f"      Kép: {recipe['images'][:60]}...")
                print(f"      Scores: HSI={recipe['HSI']:.1f}, ESI={recipe['ESI']:.1f}, PPI={recipe['PPI']:.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Mentési hiba: {e}")
            return False

def main():
    """Fő feldolgozási script"""
    processor = HungarianRecipeProcessor("hungarian_recipes_github.csv")
    
    # Teljes feldolgozás 50 recepttel a user study-hoz
    success = processor.process_all(
        output_path="data/processed_recipes.csv",
        sample_size=50
    )
    
    if success:
        print("\n🎉 FELDOLGOZÁS SIKERES!")
        print("\n📋 Következő lépések:")
        print("1. A processed_recipes.csv tartalmazza a feldolgozott recepteket")
        print("2. A user study automatikusan használni fogja a valós recepteket")
        print("3. A külső képek URL-jei megjelennek a weboldalon")
        print("4. Precision/Recall/F1 metrikák számítása implementálásra kerül")
    else:
        print("\n❌ FELDOLGOZÁS SIKERTELEN!")
        print("Ellenőrizd a 'hungarian_recipes_github.csv' fájl elérhetőségét és struktúráját.")

if __name__ == "__main__":
    main()
