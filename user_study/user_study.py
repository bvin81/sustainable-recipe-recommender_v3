#!/usr/bin/env python3
"""
EGYSZERŰSÍTETT User Study - Core funkciók
Csak a legfontosabb funkciók, tisztán és modulárisan
"""

import os
import sys
import sqlite3
import datetime
import random
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify

# Scikit-learn importok hibakezeléssel
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scikit-learn nem elérhető - fallback módban működik")
    SKLEARN_AVAILABLE = False

# Project setup
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

# Blueprint
user_study_bp = Blueprint('user_study', __name__, 
                         template_folder='templates',
                         static_folder='static')

# =============================================================================
# 1. EGYSZERŰ ADATBÁZIS OSZTÁLY
# =============================================================================

class SimpleDatabase:
    """Egyszerűsített adatbázis kezelő"""
    
    def __init__(self):
        self.db_path = data_dir / "user_study.db"
        self._init_db()
    
    def _init_db(self):
        """Adatbázis inicializálása"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS participants (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                age_group TEXT, education TEXT, cooking_frequency TEXT,
                sustainability_awareness INTEGER, version TEXT,
                is_completed BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, recipe_id INTEGER, rating INTEGER,
                explanation_helpful INTEGER, view_time_seconds REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS questionnaire (
                user_id INTEGER PRIMARY KEY,
                system_usability INTEGER, recommendation_quality INTEGER,
                trust_level INTEGER, explanation_clarity INTEGER,
                sustainability_importance INTEGER, overall_satisfaction INTEGER,
                additional_comments TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, age_group, education, cooking_frequency, sustainability_awareness, version):
        """Új felhasználó létrehozása"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            INSERT INTO participants (age_group, education, cooking_frequency, 
                                    sustainability_awareness, version)
            VALUES (?, ?, ?, ?, ?)
        ''', (age_group, education, cooking_frequency, sustainability_awareness, version))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    
    def log_interaction(self, user_id, recipe_id, rating, explanation_helpful, view_time):
        """Interakció naplózása"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO interactions (user_id, recipe_id, rating, explanation_helpful, view_time_seconds)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, recipe_id, rating, explanation_helpful, view_time))
        conn.commit()
        conn.close()
    
    def save_questionnaire(self, user_id, responses):
        """Kérdőív mentése"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO questionnaire 
            (user_id, system_usability, recommendation_quality, trust_level, 
             explanation_clarity, sustainability_importance, overall_satisfaction, additional_comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, responses['system_usability'], responses['recommendation_quality'],
              responses['trust_level'], responses['explanation_clarity'], 
              responses['sustainability_importance'], responses['overall_satisfaction'],
              responses['additional_comments']))
        
        # Mark completed
        conn.execute('UPDATE participants SET is_completed = 1 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()

# =============================================================================
# 2. EGYSZERŰ CSV KEZELŐ
# =============================================================================

class SimpleCSVLoader:
    """Egyszerűsített CSV betöltő"""
    
    @staticmethod
    def load_or_create_csv():
        """CSV betöltése vagy sample adatok generálása"""
        processed_path = data_dir / "processed_recipes.csv"
        
        # Ha létezik, betöltjük
        if processed_path.exists():
            try:
                df = pd.read_csv(processed_path)
                if len(df) >= 10:
                    print(f"✅ CSV betöltve: {len(df)} recept")
                    return df
            except:
                pass
        
        # Sample adatok generálása
        print("🔧 Sample receptek generálása...")
        sample_data = []
        
        recipes = [
            ("Gulyásleves", "marhahús, hagyma, paprika, paradicsom, burgonya", "Levesek"),
            ("Vegetáriánus Lecsó", "paprika, paradicsom, hagyma, tojás, tofu", "Vegetáriánus"),
            ("Halászlé", "ponty, csuka, hagyma, paradicsom, paprika", "Halételek"),
            ("Túrós Csusza", "széles metélt, túró, tejföl, szalonna", "Tésztaételek"),
            ("Gombapaprikás", "gomba, hagyma, paprika, tejföl, liszt", "Vegetáriánus"),
            ("Schnitzel", "sertéshús, liszt, tojás, zsemlemorzsa", "Húsételek"),
            ("Töltött Káposzta", "savanyú káposzta, darált hús, rizs", "Húsételek"),
            ("Rántott Sajt", "trappista sajt, liszt, tojás, zsemlemorzsa", "Vegetáriánus"),
            ("Babgulyás", "bab, hagyma, paprika, kolbász", "Levesek"),
            ("Palócleves", "bárány, bab, burgonya, tejföl, kapor", "Levesek")
        ]
        
        # 50 receptre bővítés
        for i in range(50):
            base_recipe = recipes[i % len(recipes)]
            recipe_id = i + 1
            title = f"{base_recipe[0]}" + (f" - {i//len(recipes) + 1}. változat" if i >= len(recipes) else "")
            
            # Random pontszámok
            np.random.seed(42 + i)
            esi = max(10, min(100, np.random.normal(65, 15)))
            hsi = max(20, min(100, np.random.normal(70, 12)))
            ppi = max(30, min(100, np.random.normal(75, 10)))
            
            sample_data.append({
                'recipeid': recipe_id,
                'title': title,
                'ingredients': base_recipe[1],
                'instructions': f"Főzési utasítás a {title} recepthez.",
                'category': base_recipe[2],
                'images': f'https://images.unsplash.com/photo-154759218{i%10}-85f173990554?w=400&h=300&fit=crop',
                'ESI': round(esi, 2),
                'HSI': round(hsi, 2),
                'PPI': round(ppi, 2),
                'composite_score': round(esi * 0.4 + hsi * 0.4 + ppi * 0.2, 2)
            })
        
        df = pd.DataFrame(sample_data)
        df.to_csv(processed_path, index=False, encoding='utf-8')
        print(f"✅ Sample CSV mentve: {len(df)} recept")
        return df

# =============================================================================
# 3. EGYSZERŰ AJÁNLÓRENDSZER
# =============================================================================

class SimpleRecommender:
    """Egyszerűsített ajánlórendszer A/B/C teszteléssel"""
    
    def __init__(self):
        self.recipes_df = SimpleCSVLoader.load_or_create_csv()
        self.search_enabled = SKLEARN_AVAILABLE
        
        if self.search_enabled:
            self._init_search()
    
    def _init_search(self):
        """Keresés inicializálása ha scikit-learn elérhető"""
        try:
            # Egyszerű TF-IDF keresés
            self.vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
            self.recipes_df['ingredients_clean'] = self.recipes_df['ingredients'].str.lower()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['ingredients_clean'])
            print("✅ Keresés inicializálva")
        except Exception as e:
            print(f"⚠️ Keresés inicializálási hiba: {e}")
            self.search_enabled = False
    
    def search_recipes(self, search_query, max_results=20):
        """Egyszerű keresés"""
        if not self.search_enabled or not search_query.strip():
            return list(range(min(max_results, len(self.recipes_df))))
        
        try:
            # TF-IDF similarity keresés
            query_vector = self.vectorizer.transform([search_query.lower()])
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[-max_results:][::-1]
            
            # Fallback: egyszerű szöveges keresés
            if similarity_scores.max() < 0.1:
                search_terms = search_query.lower().split(',')
                matching_indices = []
                for idx, ingredients in enumerate(self.recipes_df['ingredients_clean']):
                    if any(term.strip() in ingredients for term in search_terms):
                        matching_indices.append(idx)
                return matching_indices[:max_results]
            
            return [idx for idx in top_indices if similarity_scores[idx] > 0.05]
            
        except Exception as e:
            print(f"Keresési hiba: {e}")
            return list(range(min(max_results, len(self.recipes_df))))
    
    def get_recommendations(self, version='v1', search_query="", n_recommendations=5):
        """Fő ajánlási algoritmus A/B/C teszteléssel"""
        
        # 1. Keresés vagy teljes lista
        if search_query.strip():
            candidate_indices = self.search_recipes(search_query, max_results=20)
            candidates = self.recipes_df.iloc[candidate_indices].copy()
        else:
            candidates = self.recipes_df.copy()
        
        if len(candidates) == 0:
            candidates = self.recipes_df.head(n_recommendations)
        
        # 2. Pontszám alapú rendezés
        candidates['recommendation_score'] = (
            candidates['ESI'] * 0.4 +      # 40% környezeti
            candidates['HSI'] * 0.4 +      # 40% egészség  
            candidates['PPI'] * 0.2        # 20% népszerűség
        )
        
        # 3. Top N kiválasztása
        top_recipes = candidates.nlargest(n_recommendations, 'recommendation_score')
        recommendations = top_recipes.to_dict('records')
        
        # 4. A/B/C verzió-specifikus információ
        for rec in recommendations:
            if version == 'v1':
                rec['show_scores'] = False
                rec['show_explanation'] = False
                rec['explanation'] = ""
            elif version == 'v2':
                rec['show_scores'] = True
                rec['show_explanation'] = False
                rec['explanation'] = ""
            elif version == 'v3':
                rec['show_scores'] = True
                rec['show_explanation'] = True
                rec['explanation'] = self._generate_explanation(rec)
        
        return recommendations
    
    def _generate_explanation(self, recipe):
        """Egyszerű magyarázat generálás v3-hoz"""
        composite = recipe.get('composite_score', 70)
        esi = recipe.get('ESI', 70)
        
        explanation = f"Ezt a receptet {composite:.1f}/100 összpontszám alapján ajánljuk "
        explanation += "(40% környezeti + 40% egészség + 20% népszerűség). "
        
        if esi >= 80:
            explanation += "🌱 Kiváló környezeti értékeléssel"
        elif esi >= 60:
            explanation += "🌱 Környezetbarát"
        else:
            explanation += "🔸 Közepes környezeti hatással"
        
        return explanation

# =============================================================================
# 4. GLOBÁLIS OBJEKTUMOK
# =============================================================================

db = SimpleDatabase()
recommender = SimpleRecommender()

def get_user_version():
    """A/B/C verzió kiválasztása"""
    if 'version' not in session:
        session['version'] = random.choice(['v1', 'v2', 'v3'])
    return session['version']

# =============================================================================
# 5. FŐ ROUTE-OK
# =============================================================================

@user_study_bp.route('/')
def welcome():
    return render_template('welcome.html')

@user_study_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            age_group = request.form.get('age_group')
            education = request.form.get('education')
            cooking_frequency = request.form.get('cooking_frequency')
            sustainability_awareness = int(request.form.get('sustainability_awareness', 3))
            
            version = get_user_version()
            user_id = db.create_user(age_group, education, cooking_frequency, 
                                   sustainability_awareness, version)
            
            session['user_id'] = user_id
            session['version'] = version
            
            return redirect(url_for('user_study.instructions'))
            
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template('register.html', error='Regisztráció sikertelen')
    
    return render_template('register.html')

@user_study_bp.route('/instructions')
def instructions():
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    return render_template('instructions.html', version=version)

@user_study_bp.route('/study')
def study():
    """Fő tanulmány oldal"""
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    search_query = request.args.get('search', '').strip()
    
    # Ajánlások lekérése
    recommendations = recommender.get_recommendations(
        version=version, 
        search_query=search_query,
        n_recommendations=5
    )
    
    return render_template('study.html', 
                         recommendations=recommendations, 
                         version=version,
                         search_term=search_query)

@user_study_bp.route('/rate_recipe', methods=['POST'])
def rate_recipe():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    data = request.get_json()
    
    recipe_id = int(data.get('recipe_id'))
    rating = int(data.get('rating'))
    explanation_helpful = data.get('explanation_helpful')
    view_time = data.get('view_time_seconds', 0)
    
    db.log_interaction(user_id, recipe_id, rating, explanation_helpful, view_time)
    
    return jsonify({'status': 'success'})

@user_study_bp.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    if request.method == 'POST':
        user_id = session['user_id']
        
        responses = {
            'system_usability': request.form.get('system_usability'),
            'recommendation_quality': request.form.get('recommendation_quality'),
            'trust_level': request.form.get('trust_level'),
            'explanation_clarity': request.form.get('explanation_clarity'),
            'sustainability_importance': request.form.get('sustainability_importance'),
            'overall_satisfaction': request.form.get('overall_satisfaction'),
            'additional_comments': request.form.get('additional_comments', '')
        }
        
        db.save_questionnaire(user_id, responses)
        return redirect(url_for('user_study.thank_you'))
    
    version = session.get('version', 'v1')
    return render_template('questionnaire.html', version=version)

@user_study_bp.route('/thank_you')
def thank_you():
    version = session.get('version', 'v1')
    return render_template('thank_you.html', version=version)

# =============================================================================
# 6. DEBUG ROUTE-OK (egyszerűsítve)
# =============================================================================

@user_study_bp.route('/debug/status')
def debug_status():
    """Egyszerű debug státusz"""
    result = "<h2>🔍 System Status</h2>"
    
    result += f"<h3>📊 Basic Info:</h3>"
    result += f"Python: {sys.version_info.major}.{sys.version_info.minor}<br>"
    result += f"Recipes loaded: {len(recommender.recipes_df)}<br>"
    result += f"Search enabled: {'✅' if recommender.search_enabled else '❌'}<br>"
    result += f"Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}<br>"
    
    result += f"<h3>🧪 Test Recommendation:</h3>"
    try:
        test_recs = recommender.get_recommendations('v3', '', 2)
        result += f"Generated: {len(test_recs)} recommendations<br>"
        if test_recs:
            result += f"First: {test_recs[0]['title']}<br>"
            result += f"Explanation: {test_recs[0].get('explanation', 'None')}<br>"
    except Exception as e:
        result += f"Error: {e}<br>"
    
    return result

@user_study_bp.route('/debug/similarity_test')
def test_similarity():
    """Egyszerűsített similarity teszt"""
    result = "<h2>🧪 Search Test</h2>"
    
    test_queries = ["hagyma, paprika", "csirke", "gomba"]
    
    for query in test_queries:
        result += f"<h3>Keresés: '{query}'</h3>"
        
        try:
            recs = recommender.get_recommendations('v1', query, 3)
            result += f"<p>Találatok ({len(recs)}):</p><ul>"
            
            for i, rec in enumerate(recs):
                result += f"<li>{rec['title']}</li>"
            
            result += "</ul>"
            
        except Exception as e:
            result += f"<p>Hiba: {e}</p>"
        
        result += "<hr>"
    
    return result

# Export
__all__ = ['user_study_bp']
