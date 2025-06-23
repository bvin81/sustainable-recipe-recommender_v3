#!/usr/bin/env python3
"""
TELJES MEGOLDÁS - User Study with CSV Processing + Images
A `processed_recipes.csv` létrehozása és használata
"""

import os
import sys
import sqlite3
import datetime
import random
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, request, session, redirect, url_for, make_response, jsonify
# Add these imports at the top if not already present
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

class HybridRecipeRecommender:
    """Hibrid ajánlórendszer: keresés + content filtering + JAVÍTOTT similarity"""
    
    def __init__(self, csv_path):
        self.recipes_df = pd.read_csv(csv_path)
        self.count_vectorizer = None  # ÚJ: CountVectorizer
        self.tfidf_vectorizer = None  # Megtartjuk fallback-nak
        self.count_matrix = None      # ÚJ: CountVectorizer matrix
        self.tfidf_matrix = None      # Megtartjuk fallback-nak
        self.ingredient_index = None
        self._prepare_content_features()
        
    def _prepare_content_features(self):
        """Content filtering előkészítése - JAVÍTOTT verzió"""
        print("🔧 Content features előkészítése (COUNT + TF-IDF)...")
        
        # Összetevők szöveg tisztítása és normalizálása
        self.recipes_df['ingredients_clean'] = self.recipes_df['ingredients'].apply(
            self._clean_ingredients
        )
        
        # 1. COUNT VECTORIZER (ÚJ - elsődleges)
        try:
            self.count_vectorizer = CountVectorizer(
                max_features=800,        # Több feature mint TF-IDF
                stop_words=None,
                ngram_range=(1, 2),      # Unigrams + bigrams
                min_df=1,                # Minden előforduló szót megtart
                binary=True              # Binary counting (van/nincs)
            )
            
            self.count_matrix = self.count_vectorizer.fit_transform(
                self.recipes_df['ingredients_clean']
            )
            
            print(f"✅ CountVectorizer: {self.count_matrix.shape[1]} features")
            
        except Exception as e:
            print(f"⚠️ CountVectorizer hiba: {e}")
            self.count_vectorizer = None
            self.count_matrix = None
        
        # 2. TF-IDF VECTORIZER (fallback)
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=1
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.recipes_df['ingredients_clean']
            )
            
            print(f"✅ TfidfVectorizer: {self.tfidf_matrix.shape[1]} features (fallback)")
            
        except Exception as e:
            print(f"❌ TfidfVectorizer hiba: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        
        # Összetevő index építése gyors kereséshez
        self._build_ingredient_index()
        
        print(f"✅ {len(self.recipes_df)} recept feldolgozva hibrid filtering-hez")
    
    def search_by_ingredients(self, search_ingredients, max_results=20):
        """Összetevő alapú keresés - JAVÍTOTT similarity"""
        if not search_ingredients.strip():
            return list(range(min(max_results, len(self.recipes_df))))
        
        print(f"🔍 Keresés: '{search_ingredients}' (JAVÍTOTT similarity)")
        
        # 1. EXACT MATCH keresés (gyors)
        relevant_recipes = []
        ingredient_matches = {}
        
        search_terms = [term.strip().lower() for term in search_ingredients.split(',') if term.strip()]
        
        for term in search_terms:
            if term in self.ingredient_index:
                for recipe_idx in self.ingredient_index[term]:
                    if recipe_idx not in relevant_recipes:
                        relevant_recipes.append(recipe_idx)
                    if recipe_idx not in ingredient_matches:
                        ingredient_matches[recipe_idx] = 0
                    ingredient_matches[recipe_idx] += 1
        
        # 2. SIMILARITY SEARCH (CountVectorizer vagy TF-IDF)
        if not relevant_recipes or len(relevant_recipes) < max_results:
            print("🧮 Similarity search aktiválva...")
            
            # Elsődleges: CountVectorizer
            if self.count_vectorizer and self.count_matrix is not None:
                similarity_results = self._count_similarity_search(search_ingredients, max_results)
                print(f"   CountVectorizer: {len(similarity_results)} találat")
            
            # Fallback: TF-IDF
            elif self.tfidf_vectorizer and self.tfidf_matrix is not None:
                similarity_results = self._tfidf_search(search_ingredients, max_results)
                print(f"   TF-IDF fallback: {len(similarity_results)} találat")
            
            else:
                print("   ❌ Nincs elérhető similarity method")
                similarity_results = []
            
            # Kombinálás
            for idx in similarity_results:
                if idx not in relevant_recipes:
                    relevant_recipes.append(idx)
        
        # 3. RENDEZÉS és VISSZAADÁS
        if ingredient_matches:
            # Exact match-ek előre
            sorted_recipes = sorted(
                relevant_recipes, 
                key=lambda x: ingredient_matches.get(x, 0), 
                reverse=True
            )
        else:
            sorted_recipes = relevant_recipes
        
        final_results = sorted_recipes[:max_results]
        print(f"✅ Végső eredmény: {len(final_results)} recept")
        
        return final_results
    
    def _count_similarity_search(self, search_query, max_results=20):
        """CountVectorizer alapú similarity keresés - ÚJ!"""
        try:
            # Keresési lekérdezés tisztítása és vektorizálása
            query_clean = self._clean_ingredients(search_query)
            query_vector = self.count_vectorizer.transform([query_clean])
            
            # Cosine similarity számítása
            similarity_scores = cosine_similarity(query_vector, self.count_matrix).flatten()
            
            # Top eredmények kiválasztása
            top_indices = similarity_scores.argsort()[-max_results:][::-1]
            
            # Threshold: csak 0.1 feletti hasonlóságokat
            filtered_indices = [idx for idx in top_indices if similarity_scores[idx] > 0.1]
            
            print(f"   CountVectorizer scores: {[f'{similarity_scores[i]:.3f}' for i in filtered_indices[:5]]}")
            
            return filtered_indices
            
        except Exception as e:
            print(f"   ❌ CountVectorizer search hiba: {e}")
            return []
    
    def _tfidf_search(self, search_query, max_results=20):
        """TF-IDF alapú keresés - EREDETI (fallback)"""
        try:
            query_clean = self._clean_ingredients(search_query)
            query_vector = self.tfidf_vectorizer.transform([query_clean])
            
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[-max_results:][::-1]
            
            filtered_indices = [idx for idx in top_indices if similarity_scores[idx] > 0.05]
            
            print(f"   TF-IDF scores: {[f'{similarity_scores[i]:.3f}' for i in filtered_indices[:5]]}")
            
            return filtered_indices
            
        except Exception as e:
            print(f"   ❌ TF-IDF search hiba: {e}")
            return []
    
    def get_recommendations(self, version='v1', search_ingredients="", user_preferences=None, n_recommendations=5):
        """EGYSÉGES ajánlási algoritmus - csak információ megjelenítés különbözik"""
        
        # 1. KERESÉS ALAPÚ SZŰRÉS (JAVÍTOTT similarity-vel)
        if search_ingredients.strip():
            candidate_indices = self.search_by_ingredients(search_ingredients, max_results=20)
            candidate_recipes = self.recipes_df.iloc[candidate_indices].copy()
            print(f"🔍 Keresés '{search_ingredients}' -> {len(candidate_recipes)} találat")
        else:
            candidate_recipes = self.recipes_df.copy()
            print(f"📊 Teljes adatbázis -> {len(candidate_recipes)} recept")
        
        if len(candidate_recipes) == 0:
            print("❌ Nincs találat a keresésre")
            return []
        
        # 2. EGYSÉGES SCORING (minden verzióban UGYANAZ)
        search_boost = self._calculate_search_boost(candidate_recipes, search_ingredients)
        candidate_recipes['recommendation_score'] = (
            candidate_recipes['ESI'] * 0.4 +        # 40% környezeti
            candidate_recipes['HSI'] * 0.4 +        # 40% egészség
            candidate_recipes['PPI'] * 0.2 +        # 20% népszerűség
            search_boost * 0.1                      # 10% keresési relevancia
        )
        
        # 3. EGYSÉGES KIVÁLASZTÁS (minden verzióban UGYANAZ)
        final_recommendations = candidate_recipes.nlargest(n_recommendations, 'recommendation_score')
        recommendations = final_recommendations.to_dict('records')
        
        # 4. VERZIÓ-SPECIFIKUS INFORMÁCIÓ DISCLOSURE (változatlan)
        for rec in recommendations:
            rec['search_relevance'] = self._calculate_search_relevance(rec, search_ingredients)
            
            # A/B/C különbségek CSAK az információ megjelenítésében
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
                rec['explanation'] = self._generate_explanation(rec, search_ingredients)
        
        print(f"✅ {len(recommendations)} ajánlás generálva ({version}) - JAVÍTOTT similarity")
        return recommendations
    

class EnhancedRecipeRecommender:
    """Hibrid recept ajánló rendszer - EGYSÉGES ALGORITMUS + A/B/C TESTING"""
    
    def __init__(self):
        # CSV létrehozása/ellenőrzése
        self.csv_path = CSVProcessor.create_processed_csv()
        self.recipes_df = self.load_recipes()
        
        # Hibrid rendszer inicializálása
        if self.recipes_df is not None:
            try:
                self.hybrid_recommender = HybridRecipeRecommender(str(self.csv_path))
                print(f"🍽️ Hibrid ajánló rendszer inicializálva: {len(self.recipes_df)} recept")
            except Exception as e:
                print(f"⚠️ Hibrid ajánló inicializálási hiba: {e}")
                self.hybrid_recommender = None
        else:
            self.hybrid_recommender = None
    
    def load_recipes(self):
        """Receptek betöltése CSV-ből"""
        try:
            if not self.csv_path.exists():
                print(f"❌ CSV nem található: {self.csv_path}")
                return None
            
            df = pd.read_csv(self.csv_path)
            print(f"✅ CSV betöltve: {len(df)} recept")
            
            # Kötelező oszlopok ellenőrzése
            required_cols = ['recipeid', 'title', 'ingredients', 'images', 'HSI', 'ESI', 'PPI']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Hiányzó oszlopok: {missing_cols}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ CSV betöltési hiba: {e}")
            return None
    
    def get_recommendations(self, version='v1', search_ingredients="", user_preferences=None, n_recommendations=5):
        """HIBRID ajánlások lekérése - A/B/C TESTING"""
        if self.hybrid_recommender is None:
            print("❌ Hibrid ajánló nem elérhető! Fallback...")
            return self._fallback_recommendations(version, n_recommendations)
        
        try:
            # Felhasználói preferenciák session-ből
            if user_preferences is None:
                user_preferences = {}
            
            # Hibrid ajánló hívása
            recommendations = self.hybrid_recommender.get_recommendations(
                version=version,
                search_ingredients=search_ingredients,
                user_preferences=user_preferences,
                n_recommendations=n_recommendations
            )
            
            print(f"✅ {len(recommendations)} hibrid ajánlás generálva ({version})")
            return recommendations
            
        except Exception as e:
            print(f"❌ Hibrid ajánlási hiba: {e}")
            return self._fallback_recommendations(version, n_recommendations)
    
   # user_study.py - Cseréld ki a _fallback_recommendations metódust az EnhancedRecipeRecommender osztályban

def _fallback_recommendations(self, version, n_recommendations):
    """Fallback ajánlások ha a hibrid rendszer nem működik"""
    print(f"⚠️ FALLBACK MODE: Generating {n_recommendations} sample recommendations for {version}")
    
    try:
        # Ha van betöltött CSV, használjuk azt
        if self.recipes_df is not None and len(self.recipes_df) > 0:
            print(f"📊 Using CSV data: {len(self.recipes_df)} recipes available")
            sample_size = min(n_recommendations, len(self.recipes_df))
            recommendations = self.recipes_df.sample(n=sample_size).to_dict('records')
        else:
            # Ha nincs CSV, generálj sample adatokat
            print("🔧 Generating hardcoded fallback recipes")
            sample_recipes = [
                {
                    'recipeid': 1,
                    'title': 'Hagyományos Gulyásleves',
                    'ingredients': 'marhahús, hagyma, paprika, paradicsom, burgonya, fokhagyma, kömény, majoranna',
                    'instructions': 'A húst kockákra vágjuk és enyhén megsózzuk. Megdinszteljük a hagymát, hozzáadjuk a paprikát. Felöntjük vízzel és főzzük 1.5 órát.',
                    'images': 'https://images.unsplash.com/photo-1547592180-85f173990554?w=400&h=300&fit=crop',
                    'HSI': 75.0, 'ESI': 60.0, 'PPI': 90.0, 'composite_score': 71.0
                },
                {
                    'recipeid': 2,
                    'title': 'Vegetáriánus Lecsó',
                    'ingredients': 'paprika, paradicsom, hagyma, tojás, tofu, olívaolaj, só, bors, fokhagyma',
                    'instructions': 'A hagymát és fokhagymát megdinszteljük olívaolajban. Hozzáadjuk a felszeletelt paprikát.',
                    'images': 'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400&h=300&fit=crop',
                    'HSI': 85.0, 'ESI': 80.0, 'PPI': 70.0, 'composite_score': 78.0
                },
                {
                    'recipeid': 3,
                    'title': 'Halászlé Szegedi Módra',
                    'ingredients': 'ponty, csuka, harcsa, hagyma, paradicsom, paprika, só, babérlevél',
                    'instructions': 'A halakat megtisztítjuk és feldaraboljuk. A halak fejéből és farkából erős alapot főzünk.',
                    'images': 'https://images.unsplash.com/photo-1544943910-4c1dc44aab44?w=400&h=300&fit=crop',
                    'HSI': 80.0, 'ESI': 70.0, 'PPI': 75.0, 'composite_score': 74.0
                },
                {
                    'recipeid': 4,
                    'title': 'Túrós Csusza',
                    'ingredients': 'széles metélt, túró, tejföl, szalonna, hagyma, só, bors',
                    'instructions': 'A tésztát sós vízben megfőzzük és leszűrjük. A szalonnát kockákra vágjuk és kisütjük.',
                    'images': 'https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=300&fit=crop',
                    'HSI': 65.0, 'ESI': 55.0, 'PPI': 80.0, 'composite_score': 65.0
                },
                {
                    'recipeid': 5,
                    'title': 'Gombapaprikás Galuskával',
                    'ingredients': 'gomba, hagyma, paprika, tejföl, liszt, tojás, petrezselyem, olaj',
                    'instructions': 'A gombát felszeleteljük és kisütjük. Hagymát dinsztelünk, paprikát adunk hozzá.',
                    'images': 'https://images.unsplash.com/photo-1565299507177-b0ac66763828?w=400&h=300&fit=crop',
                    'HSI': 70.0, 'ESI': 75.0, 'PPI': 65.0, 'composite_score': 70.0
                }
            ]
            
            # Válassz ki annyit amennyit kértek
            recommendations = sample_recipes[:n_recommendations]
        
        # Verzió-specifikus információ hozzáadása MINDEN recepthez
        for rec in recommendations:
            # Biztosítsd hogy minden szükséges mező létezik
            if 'HSI' not in rec:
                rec['HSI'] = 70.0
            if 'ESI' not in rec:
                rec['ESI'] = 75.0
            if 'PPI' not in rec:
                rec['PPI'] = 80.0
            if 'composite_score' not in rec:
                rec['composite_score'] = (rec['ESI'] * 0.4 + rec['HSI'] * 0.4 + rec['PPI'] * 0.2)
            
            # A/B/C testing verzió-specifikus megjelenítés
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
                rec['explanation'] = f"Ezt a receptet {rec['composite_score']:.1f}/100 összpontszám alapján ajánljuk (40% környezeti + 40% egészség + 20% népszerűség). Fallback módban működik."
            
            # Search relevance fallback
            rec['search_relevance'] = 0.0
        
        print(f"✅ Fallback recommendations generated: {len(recommendations)} recipes for version {version}")
        return recommendations
        
    except Exception as e:
        print(f"❌ CRITICAL FALLBACK ERROR: {e}")
        # Ultimate fallback - egyetlen minimal recept
        minimal_recipe = {
            'recipeid': 1,
            'title': 'Alaprecept (Rendszer helyreállítás alatt)',
            'ingredients': 'Alapösszetevők',
            'instructions': 'Alapinstrukciók',
            'images': 'https://via.placeholder.com/400x300/cccccc/666666?text=Recipe',
            'HSI': 70.0,
            'ESI': 70.0,
            'PPI': 70.0,
            'composite_score': 70.0,
            'show_scores': version != 'v1',
            'show_explanation': version == 'v3',
            'explanation': 'Rendszer helyreállítás alatt.' if version == 'v3' else '',
            'search_relevance': 0.0
        }
        return [minimal_recipe]
        return []

# Global objektumok
db = UserStudyDatabase()
recommender = EnhancedRecipeRecommender()

def get_user_version():
    if 'version' not in session:
        versions = ['v1', 'v2', 'v3']
        session['version'] = random.choice(versions)
    return session['version']

# ROUTES

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
    """Fő tanulmány oldal - A/B/C TESTING + HIBRID KERESÉS"""
    if 'user_id' not in session:
        return redirect(url_for('user_study.register'))
    
    version = session.get('version', 'v1')
    
    # Keresési paraméter
    search_ingredients = request.args.get('search', '').strip()
    
    # Felhasználói preferenciák session-ből
    user_preferences = {
        'sustainability_awareness': session.get('sustainability_awareness', 3),
        'cooking_frequency': session.get('cooking_frequency', ''),
        'education': session.get('education', '')
    }
    
    # HIBRID ajánlások lekérése - EGYSÉGES ALGORITMUS
    recommendations = recommender.get_recommendations(
        version=version, 
        search_ingredients=search_ingredients,
        user_preferences=user_preferences,
        n_recommendations=5
    )
    
    if not recommendations:
        return "❌ Hiba: Nem sikerült betölteni a recepteket. Próbálja újra később.", 500
    
    print(f"🔍 Template-nek átadott {len(recommendations)} ajánlás ({version}) - Keresés: '{search_ingredients}'")
    
    return render_template('study.html', 
                         recommendations=recommendations, 
                         version=version,
                         search_term=search_ingredients)

# Add ingredient suggestions API
@user_study_bp.route('/api/ingredient_suggestions')
def ingredient_suggestions():
    """Összetevő javaslatok API"""
    try:
        partial_input = request.args.get('q', '').strip()
        
        if len(partial_input) < 2:
            return jsonify([])
        
        if recommender.hybrid_recommender:
            suggestions = recommender.hybrid_recommender.get_ingredient_suggestions(partial_input)
            return jsonify(suggestions)
        else:
            return jsonify([])
            
    except Exception as e:
        print(f"Suggestion API error: {e}")
        return jsonify([])

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
    interaction_order = data.get('interaction_order', 0)
    
    db.log_interaction(user_id, recipe_id, rating, explanation_helpful, view_time, interaction_order)
    
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

@user_study_bp.route('/admin/stats')
def admin_stats():
    """Admin statisztikák"""
    try:
        conn = db.get_connection()
        
        stats = {}
        
        # Alapstatisztikák
        result = conn.execute('SELECT COUNT(*) as count FROM participants').fetchone()
        stats['total_participants'] = result['count'] if result else 0
        
        result = conn.execute('SELECT COUNT(*) as count FROM participants WHERE is_completed = 1').fetchone()
        stats['completed_participants'] = result['count'] if result else 0
        
        if stats['total_participants'] > 0:
            stats['completion_rate'] = stats['completed_participants'] / stats['total_participants']
        else:
            stats['completion_rate'] = 0
            
        
        # Verzió eloszlás
        version_results = conn.execute('''
            SELECT version, 
                   COUNT(*) as count,
                   SUM(CASE WHEN is_completed = 1 THEN 1 ELSE 0 END) as completed
            FROM participants 
            GROUP BY version
        ''').fetchall()
        
        stats['version_distribution'] = [dict(row) for row in version_results]
        
        # Átlagos értékelések
        rating_results = conn.execute('''
            SELECT p.version, AVG(i.rating) as avg_rating, COUNT(i.rating) as count
            FROM participants p
            JOIN interactions i ON p.user_id = i.user_id
            WHERE i.rating IS NOT NULL
            GROUP BY p.version
        ''').fetchall()
        
        stats['average_ratings'] = [dict(row) for row in rating_results]
        
        # Kérdőív eredmények
        questionnaire_results = conn.execute('''
            SELECT p.version,
                   AVG(q.system_usability) as avg_usability,
                   AVG(q.recommendation_quality) as avg_quality,
                   AVG(q.trust_level) as avg_trust,
                   AVG(q.explanation_clarity) as avg_clarity,
                   AVG(q.overall_satisfaction) as avg_satisfaction
            FROM participants p
            JOIN questionnaire q ON p.user_id = q.user_id
            GROUP BY p.version
        ''').fetchall()
        
        stats['questionnaire_results'] = [dict(row) for row in questionnaire_results]
        
        # Átlagos interakciók
        interactions_count = conn.execute('SELECT COUNT(*) as count FROM interactions').fetchone()
        if stats['total_participants'] > 0:
            stats['avg_interactions_per_user'] = interactions_count['count'] / stats['total_participants']
        else:
            stats['avg_interactions_per_user'] = 0
        
        conn.close()
        
        return render_template('admin_stats.html', stats=stats)
        
    except Exception as e:
        return f"Stats error: {e}", 500
# Add these routes to user_study.py after the admin_stats route

@user_study_bp.route('/admin/export/csv')
def export_csv():
    """CSV export SPSS/Excel kompatibilis formátumban"""
    try:
        from flask import make_response
        import io
        import csv
        
        conn = db.get_connection()
        
        # Összevont adatok lekérése
        query = '''
        SELECT 
            p.user_id,
            p.age_group,
            p.education,
            p.cooking_frequency,
            p.sustainability_awareness,
            p.version,
            p.is_completed,
            p.created_at as registration_time,
            q.system_usability,
            q.recommendation_quality,
            q.trust_level,
            q.explanation_clarity,
            q.sustainability_importance,
            q.overall_satisfaction,
            q.additional_comments,
            q.timestamp as questionnaire_time,
            GROUP_CONCAT(i.recipe_id) as rated_recipes,
            GROUP_CONCAT(i.rating) as ratings,
            GROUP_CONCAT(i.explanation_helpful) as explanation_ratings,
            AVG(i.rating) as avg_rating,
            COUNT(i.rating) as total_ratings,
            AVG(i.view_time_seconds) as avg_view_time
        FROM participants p
        LEFT JOIN questionnaire q ON p.user_id = q.user_id
        LEFT JOIN interactions i ON p.user_id = i.user_id
        GROUP BY p.user_id
        ORDER BY p.user_id
        '''
        
        results = conn.execute(query).fetchall()
        conn.close()
        
        if not results:
            return "Nincs exportálható adat.", 404
        
        # CSV generálása
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = [
            'UserID', 'AgeGroup', 'Education', 'CookingFrequency', 'SustainabilityAwareness',
            'Version', 'IsCompleted', 'RegistrationTime',
            'SystemUsability', 'RecommendationQuality', 'TrustLevel', 'ExplanationClarity',
            'SustainabilityImportance', 'OverallSatisfaction', 'AdditionalComments',
            'QuestionnaireTime', 'RatedRecipes', 'Ratings', 'ExplanationRatings',
            'AvgRating', 'TotalRatings', 'AvgViewTime'
        ]
        writer.writerow(headers)
        
        # Adatok
        for row in results:
            # Version mapping numerikusra (SPSS-hez)
            version_num = {'v1': 1, 'v2': 2, 'v3': 3}.get(row['version'], 0)
            
            csv_row = [
                row['user_id'],
                row['age_group'],
                row['education'],
                row['cooking_frequency'],
                row['sustainability_awareness'],
                version_num,  # Numerikus verzió
                1 if row['is_completed'] else 0,  # Boolean -> 0/1
                row['registration_time'],
                row['system_usability'],
                row['recommendation_quality'],
                row['trust_level'],
                row['explanation_clarity'],
                row['sustainability_importance'],
                row['overall_satisfaction'],
                row['additional_comments'] or '',
                row['questionnaire_time'],
                row['rated_recipes'] or '',
                row['ratings'] or '',
                row['explanation_ratings'] or '',
                round(row['avg_rating'], 2) if row['avg_rating'] else '',
                row['total_ratings'] or 0,
                round(row['avg_view_time'], 2) if row['avg_view_time'] else ''
            ]
            writer.writerow(csv_row)
        
        # Response készítése
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=user_study_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return f"CSV export hiba: {e}", 500


@user_study_bp.route('/admin/export/json')
def export_json():
    """JSON export API/programozás célokra"""
    try:
        conn = db.get_connection()
        
        # Participants
        participants = conn.execute('SELECT * FROM participants').fetchall()
        participants_data = [dict(row) for row in participants]
        
        # Interactions
        interactions = conn.execute('SELECT * FROM interactions').fetchall()
        interactions_data = [dict(row) for row in interactions]
        
        # Questionnaire
        questionnaire = conn.execute('SELECT * FROM questionnaire').fetchall()
        questionnaire_data = [dict(row) for row in questionnaire]
        
        conn.close()
        
        # JSON struktúra
        export_data = {
            'export_info': {
                'timestamp': datetime.datetime.now().isoformat(),
                'total_participants': len(participants_data),
                'total_interactions': len(interactions_data),
                'total_questionnaires': len(questionnaire_data),
                'version': 'Sustainable Recipe Recommender v2.0'
            },
            'participants': participants_data,
            'interactions': interactions_data,
            'questionnaire': questionnaire_data,
            'summary_stats': {
                'completion_rate': len(questionnaire_data) / len(participants_data) if participants_data else 0,
                'avg_interactions_per_user': len(interactions_data) / len(participants_data) if participants_data else 0,
                'version_distribution': {}
            }
        }
        
        # Version distribution
        for participant in participants_data:
            version = participant['version']
            if version not in export_data['summary_stats']['version_distribution']:
                export_data['summary_stats']['version_distribution'][version] = 0
            export_data['summary_stats']['version_distribution'][version] += 1
        
        response = jsonify(export_data)
        response.headers['Content-Disposition'] = f'attachment; filename=user_study_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'JSON export hiba: {e}'}), 500


@user_study_bp.route('/admin/export/spss_syntax')
def export_spss_syntax():
    """SPSS syntax fájl generálása"""
    try:
        spss_syntax = '''
* SPSS Syntax for Sustainable Recipe Recommender User Study
* Generated automatically on {timestamp}

* Load CSV data
GET DATA
 /TYPE=TXT
 /FILE='user_study_data.csv'
 /ENCODING='UTF8'
 /DELIMITERS=","
 /QUALIFIER='"'
 /ARRANGEMENT=DELIMITED
 /FIRSTCASE=2
 /VARIABLES=
 UserID F8.0
 AgeGroup A20
 Education A30
 CookingFrequency A20
 SustainabilityAwareness F2.0
 Version F1.0
 IsCompleted F1.0
 RegistrationTime A25
 SystemUsability F2.0
 RecommendationQuality F2.0
 TrustLevel F2.0
 ExplanationClarity F2.0
 SustainabilityImportance F2.0
 OverallSatisfaction F2.0
 AdditionalComments A500
 QuestionnaireTime A25
 RatedRecipes A100
 Ratings A50
 ExplanationRatings A50
 AvgRating F4.2
 TotalRatings F2.0
 AvgViewTime F6.2.

* Variable labels
VARIABLE LABELS
 UserID 'Unique User Identifier'
 AgeGroup 'Age Group Category'
 Education 'Education Level'
 CookingFrequency 'Cooking Frequency'
 SustainabilityAwareness 'Sustainability Awareness (1-5)'
 Version 'System Version (1=v1, 2=v2, 3=v3)'
 IsCompleted 'Study Completed (0=No, 1=Yes)'
 SystemUsability 'System Usability Rating (1-5)'
 RecommendationQuality 'Recommendation Quality Rating (1-5)'
 TrustLevel 'Trust Level Rating (1-5)'
 ExplanationClarity 'Explanation Clarity Rating (1-5)'
 SustainabilityImportance 'Sustainability Importance Rating (1-5)'
 OverallSatisfaction 'Overall Satisfaction Rating (1-5)'
 AvgRating 'Average Recipe Rating'
 TotalRatings 'Total Number of Ratings Given'
 AvgViewTime 'Average View Time per Recipe (seconds)'.

* Value labels
VALUE LABELS Version
 1 'Baseline (v1)'
 2 'Score Disclosure (v2)'
 3 'Full Disclosure + XAI (v3)'.

VALUE LABELS IsCompleted
 0 'Not Completed'
 1 'Completed'.

* Descriptive statistics
DESCRIPTIVES VARIABLES=SystemUsability RecommendationQuality TrustLevel 
 ExplanationClarity SustainabilityImportance OverallSatisfaction
 AvgRating TotalRatings AvgViewTime
 /STATISTICS=MEAN STDDEV MIN MAX.

* Frequency analysis
FREQUENCIES VARIABLES=Version AgeGroup Education CookingFrequency IsCompleted.

* One-way ANOVA for version comparison
ONEWAY SystemUsability BY Version
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY RecommendationQuality BY Version
 /STATISTICS DESCRIPTIVES  
 /POSTHOC TUKEY.

ONEWAY TrustLevel BY Version
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY OverallSatisfaction BY Version
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

* Chi-square for categorical variables
CROSSTABS
 /TABLES=Version BY AgeGroup Education CookingFrequency
 /STATISTICS=CHISQ.

* Save processed dataset
SAVE OUTFILE='user_study_processed.sav'.
        '''.format(timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        response = make_response(spss_syntax)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=spss_analysis_{datetime.datetime.now().strftime("%Y%m%d")}.sps'
        
        return response
        
    except Exception as e:
        return f"SPSS syntax hiba: {e}", 500
# DEBUG route CSV ellenőrzéshez
@user_study_bp.route('/debug/csv')
def debug_csv():
    """CSV debug információk"""
    try:
        result = "<h2>🔍 CSV Debug Information</h2>"
        
        # Processed CSV ellenőrzés
        csv_path = project_root / "data" / "processed_recipes.csv"
        result += f"<h3>📊 Processed CSV Status:</h3>"
        result += f"Path: {csv_path}<br>"
        result += f"Exists: {'✅ YES' if csv_path.exists() else '❌ NO'}<br>"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                result += f"Rows: {len(df)}<br>"
                result += f"Columns: {list(df.columns)}<br><br>"
                
                result += "<h3>🖼️ Image URLs (first 3):</h3>"
                for i in range(min(3, len(df))):
                    recipe = df.iloc[i]
                    result += f"<b>{recipe['title']}:</b><br>"
                    result += f"Image: {recipe.get('images', 'NINCS')}<br><br>"
                
            except Exception as e:
                result += f"CSV read error: {e}<br>"
        
        # Original CSV ellenőrzés
        original_csv = project_root / "hungarian_recipes_github.csv"
        result += f"<h3>📋 Original CSV Status:</h3>"
        result += f"Path: {original_csv}<br>"
        result += f"Exists: {'✅ YES' if original_csv.exists() else '❌ NO'}<br>"
        
        if original_csv.exists():
            try:
                df_orig = pd.read_csv(original_csv)
                result += f"Rows: {len(df_orig)}<br>"
                result += f"Columns: {list(df_orig.columns)}<br>"
            except Exception as e:
                result += f"Original CSV read error: {e}<br>"
        
        # Recommender status
        result += f"<h3>🤖 Recommender Status:</h3>"
        result += f"Recipes loaded: {len(recommender.recipes_df) if recommender.recipes_df is not None else 0}<br>"
        
        # Test recommendation
        try:
            test_recs = recommender.get_recommendations('v1', 2)
            result += f"Test recommendations: {len(test_recs)}<br>"
            if test_recs:
                result += f"First recipe: {test_recs[0]['title']}<br>"
                result += f"First image: {test_recs[0].get('images', 'NINCS')}<br>"
        except Exception as e:
            result += f"Test recommendation error: {e}<br>"
        
        return result
        
    except Exception as e:
        return f"Debug error: {e}"

@user_study_bp.route('/debug/esi_zero')
def debug_esi_zero():
    """Debug ESI=0 values"""
    try:
        result = "<h2>🔍 ESI=0 Debug Analysis</h2>"
        
        # Load processed CSV
        csv_path = project_root / "data" / "processed_recipes.csv"
        if not csv_path.exists():
            return "❌ processed_recipes.csv not found"
        
        df = pd.read_csv(csv_path)
        result += f"<h3>📊 CSV Statistics:</h3>"
        result += f"Total recipes: {len(df)}<br>"
        
        # Check score columns
        score_cols = ['ESI', 'HSI', 'PPI', 'composite_score']
        for col in score_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                zero_count = (df[col] == 0).sum()
                
                result += f"<b>{col}:</b> {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f}, zeros: {zero_count})<br>"
        
        # Find recipes with ESI=0
        zero_esi_recipes = df[df['ESI'] == 0]
        if len(zero_esi_recipes) > 0:
            result += f"<h3>❌ Recipes with ESI=0 ({len(zero_esi_recipes)} found):</h3>"
            for _, recipe in zero_esi_recipes.head(5).iterrows():
                result += f"<b>{recipe['title']}:</b><br>"
                result += f"   ESI: {recipe['ESI']:.2f}<br>"
                result += f"   HSI: {recipe['HSI']:.2f}<br>"
                result += f"   PPI: {recipe['PPI']:.2f}<br>"
                result += f"<br>"
        else:
            result += f"<h3>✅ No recipes with ESI=0 found</h3>"
        
        return result
        
    except Exception as e:
        return f"Debug error: {e}"
        
@user_study_bp.route('/debug/abc_testing')
def debug_abc_testing():
    """Debug A/B/C testing működését"""
    try:
        result = "<h2>🧪 A/B/C Testing Debug</h2>"
        
        # Check recommender status
        result += f"<h3>🔧 Recommender Status:</h3>"
        result += f"<p><strong>Recommender type:</strong> {type(recommender).__name__}</p>"
        result += f"<p><strong>Has hybrid_recommender:</strong> {hasattr(recommender, 'hybrid_recommender')}</p>"
        
        if hasattr(recommender, 'hybrid_recommender'):
            result += f"<p><strong>Hybrid recommender:</strong> {recommender.hybrid_recommender is not None}</p>"
            
            if recommender.hybrid_recommender:
                result += f"<p><strong>Hybrid type:</strong> {type(recommender.hybrid_recommender).__name__}</p>"
        
        # Test all three versions
        test_versions = ['v1', 'v2', 'v3']
        
        for version in test_versions:
            result += f"<h3>📊 {version.upper()} Version Test:</h3>"
            
            try:
                # Get test recommendations
                recommendations = recommender.get_recommendations(
                    version=version,
                    search_ingredients="",
                    user_preferences={},
                    n_recommendations=2
                )
                
                result += f"<p><strong>Recommendations count:</strong> {len(recommendations)}</p>"
                
                for i, rec in enumerate(recommendations):
                    result += f"<h4>Recipe {i+1}: {rec.get('title', 'NO TITLE')}</h4>"
                    result += f"<p><strong>show_scores:</strong> {rec.get('show_scores', 'MISSING')}</p>"
                    result += f"<p><strong>show_explanation:</strong> {rec.get('show_explanation', 'MISSING')}</p>"
                    result += f"<p><strong>explanation:</strong> {rec.get('explanation', 'EMPTY')}</p>"
                    result += f"<p><strong>HSI:</strong> {rec.get('HSI', 'N/A')}</p>"
                    result += "<hr>"
                    
            except Exception as e:
                result += f"<p>❌ Error testing {version}: {e}</p>"
        
        return result
        
    except Exception as e:
        return f"Debug error: {e}", 500
# Add this to user_study.py right after the other debug routes (around line 650-700)

@user_study_bp.route('/debug/emergency')
def emergency_debug():
    """Emergency debug - mi okozza a fallback módot"""
    try:
        result = "<h2>🚨 Emergency Debug</h2>"
        
        # 1. Basic system info
        result += f"<h3>📊 System Status:</h3>"
        result += f"<p><strong>Python:</strong> {sys.version}</p>"
        result += f"<p><strong>Working dir:</strong> {os.getcwd()}</p>"
        
        # 2. Check imports
        result += f"<h3>📦 Import Status:</h3>"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            result += f"<p>✅ scikit-learn: MŰKÖDIK</p>"
        except ImportError as e:
            result += f"<p>❌ scikit-learn: HIÁNYZIK - {e}</p>"
        
        try:
            import re
            result += f"<p>✅ re module: MŰKÖDIK</p>"
        except ImportError as e:
            result += f"<p>❌ re module: HIÁNYZIK - {e}</p>"
        
        # 3. Check CSV
        result += f"<h3>📊 CSV Status:</h3>"
        csv_path = project_root / "data" / "processed_recipes.csv"
        result += f"<p><strong>CSV path:</strong> {csv_path}</p>"
        result += f"<p><strong>CSV exists:</strong> {csv_path.exists()}</p>"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                result += f"<p>✅ CSV loaded: {len(df)} rows</p>"
                result += f"<p><strong>Columns:</strong> {list(df.columns)}</p>"
                
                # Show first recipe
                if len(df) > 0:
                    first_recipe = df.iloc[0]
                    result += f"<p><strong>First recipe:</strong> {first_recipe['title']}</p>"
                
            except Exception as e:
                result += f"<p>❌ CSV load error: {e}</p>"
        
        # 4. Check recommender
        result += f"<h3>🤖 Recommender Status:</h3>"
        result += f"<p><strong>Recommender type:</strong> {type(recommender).__name__}</p>"
        result += f"<p><strong>Has recipes_df:</strong> {hasattr(recommender, 'recipes_df')}</p>"
        
        if hasattr(recommender, 'recipes_df'):
            if recommender.recipes_df is not None:
                result += f"<p>✅ Recipes loaded: {len(recommender.recipes_df)} rows</p>"
            else:
                result += f"<p>❌ Recipes_df is None</p>"
        
        result += f"<p><strong>Has hybrid_recommender:</strong> {hasattr(recommender, 'hybrid_recommender')}</p>"
        
        if hasattr(recommender, 'hybrid_recommender'):
            if recommender.hybrid_recommender is not None:
                result += f"<p>✅ Hybrid recommender: INITIALIZED</p>"
                result += f"<p><strong>Hybrid type:</strong> {type(recommender.hybrid_recommender).__name__}</p>"
            else:
                result += f"<p>❌ Hybrid recommender: NULL (ezért fallback!)</p>"
        
        # 5. Test basic function
        result += f"<h3>🧪 Function Test:</h3>"
        try:
            # Próbáld meg a legegyszerűbb hívást
            test_recs = recommender.get_recommendations('v1', '', {}, 2)
            result += f"<p>✅ get_recommendations: {len(test_recs)} recipes returned</p>"
            
            if len(test_recs) > 0:
                first_rec = test_recs[0]
                result += f"<p><strong>First recipe title:</strong> {first_rec.get('title', 'NO TITLE')}</p>"
                result += f"<p><strong>Show scores:</strong> {first_rec.get('show_scores', 'MISSING')}</p>"
                result += f"<p><strong>Has explanation:</strong> {bool(first_rec.get('explanation', ''))}</p>"
                
        except Exception as e:
            result += f"<p>❌ get_recommendations ERROR: {str(e)}</p>"
            result += f"<p><strong>Error type:</strong> {type(e).__name__}</p>"
            
            # Részletes traceback
            import traceback
            result += f"<pre style='background: #f0f0f0; padding: 10px; font-size: 12px;'>{traceback.format_exc()}</pre>"
        
        # 6. Requirements check
        result += f"<h3>📋 Requirements Status:</h3>"
        try:
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            sklearn_installed = any('scikit' in pkg.lower() for pkg in installed_packages)
            result += f"<p><strong>Scikit-learn in packages:</strong> {sklearn_installed}</p>"
            
            # List some key packages
            key_packages = ['pandas', 'numpy', 'flask', 'scikit-learn']
            for pkg in key_packages:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    result += f"<p>✅ {pkg}: {version}</p>"
                except:
                    result += f"<p>❌ {pkg}: NOT FOUND</p>"
                    
        except ImportError:
            result += f"<p>⚠️ pkg_resources not available</p>"
        
        return result
        
    except Exception as e:
        import traceback
        return f"<h1>TOTAL EMERGENCY ERROR:</h1><p>{e}</p><pre>{traceback.format_exc()}</pre>"
@user_study_bp.route('/debug/similarity_test')
def test_similarity():
    """Similarity algoritmus tesztelése"""
    result = "<h2>🧪 Similarity Algorithm Test</h2>"
    
    test_queries = [
        "hagyma, paprika",
        "csirke, burgonya", 
        "tészta, paradicsom",
        "gomba, tejföl"
    ]
    
    for query in test_queries:
        result += f"<h3>Keresés: '{query}'</h3>"
        
        try:
            if recommender.hybrid_recommender:
                # Test both methods
                indices = recommender.hybrid_recommender.search_by_ingredients(query, max_results=5)
                recipes = recommender.hybrid_recommender.recipes_df.iloc[indices]
                
                result += f"<p><strong>Találatok ({len(indices)}):</strong></p><ul>"
                
                for i, (idx, recipe) in enumerate(recipes.iterrows()):
                    result += f"<li>{i+1}. {recipe.get('title', 'N/A')} <small>(ID: {idx})</small></li>"
                
                result += "</ul>"
                
                # Method info
                vectorizer_type = "CountVectorizer" if recommender.hybrid_recommender.count_vectorizer else "TF-IDF"
                result += f"<small>Használt módszer: {vectorizer_type}</small>"
                
            else:
                result += "<p>❌ Hibrid ajánló nem elérhető</p>"
                
        except Exception as e:
            result += f"<p>❌ Hiba: {e}</p>"
        
        result += "<hr>"
    
    return result

# Export
__all__ = ['user_study_bp']
