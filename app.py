#!/usr/bin/env python3
"""
Sustainable Recipe Recommender - Main Flask Application
Enhanced version with proper indentation and debug routes
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request, redirect, url_for
import pandas as pd

# Project path setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_app():
    """Enhanced Flask app létrehozása"""
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-for-development')
    
    # User study import
    try:
        from user_study.user_study import user_study_bp
        app.register_blueprint(user_study_bp)
        print("✅ User study blueprint registered")
    except ImportError as e:
        print(f"⚠️ User study import failed: {e}")
    
    @app.route('/')
    def index():
        """Főoldal - átirányítás a tanulmányhoz"""
        return redirect(url_for('user_study.welcome'))
    
    @app.route('/health')
    def health():
        """Heroku health check"""
        return jsonify({
            "status": "healthy",
            "service": "sustainable-recipe-recommender",
            "version": "2.0"
        })
    
    @app.route('/debug/system')
    def debug_system():
        """Rendszer debug információk"""
        try:
            result = "<h2>System Debug Info:</h2>"
            
            # File system ellenőrzés
            files_check = {
                'hungarian_recipes_github.csv': os.path.exists('hungarian_recipes_github.csv'),
                'recipe_preprocessor.py': os.path.exists('recipe_preprocessor.py'),
                'data/processed_recipes.csv': os.path.exists('data/processed_recipes.csv'),
                'user_study/user_study.py': os.path.exists('user_study/user_study.py')
            }
            
            result += "<h3>File System:</h3>"
            for file, exists in files_check.items():
                status = "✅" if exists else "❌"
                result += f"{status} {file}<br>"
            
            # Python environment
            result += f"<h3>Environment:</h3>"
            result += f"Python: {sys.version}<br>"
            result += f"Working directory: {os.getcwd()}<br>"
            result += f"PATH: {sys.path[:3]}<br>"
            
            return result
            
        except Exception as e:
            return f"System debug error: {e}"
    
    @app.route('/debug/recipes')
    def debug_recipes():
        """Recipe debug információk"""
        try:
            result = "<h2>Recipe Debug Info:</h2>"
            
            # Processed recipes ellenőrzés
            if os.path.exists('data/processed_recipes.csv'):
                df = pd.read_csv('data/processed_recipes.csv')
                result += f"✅ Processed recipes: {len(df)} darab<br>"
                result += f"Oszlopok: {list(df.columns)}<br><br>"
                
                result += "<h3>Első 3 recept:</h3>"
                for i in range(min(3, len(df))):
                    recipe = df.iloc[i]
                    result += f"<b>{recipe['title']}:</b><br>"
                    result += f"Kép: {recipe.get('images', 'NINCS')}<br><br>"
            else:
                result += "❌ processed_recipes.csv nem található<br>"
            
            # Original CSV ellenőrzés
            if os.path.exists('hungarian_recipes_github.csv'):
                try:
                    csv_df = pd.read_csv('hungarian_recipes_github.csv')
                    result += f"<h3>✅ Original CSV:</h3>"
                    result += f"Sorok: {len(csv_df)}<br>"
                    result += f"Oszlopok: {list(csv_df.columns)}<br>"
                    
                    if 'images' in csv_df.columns:
                        first_image = str(csv_df['images'].iloc[0])
                        result += f"Első kép minta: {first_image[:100]}...<br>"
                except Exception as e:
                    result += f"❌ CSV olvasási hiba: {e}<br>"
            else:
                result += "❌ hungarian_recipes_github.csv nem található<br>"
            
            return result
            
        except Exception as e:
            return f"Recipe debug error: {e}"
    
    return app

def create_fallback_app():
    """Fallback alkalmazás ha a fő app nem működik"""
    app = Flask(__name__)
    
    @app.route('/')
    def fallback_home():
        return """
        <html>
        <head><title>Sustainable Recipe Recommender</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🌱 Sustainable Recipe Recommender</h1>
            <h2>⚙️ System Initializing...</h2>
            <p>Az alkalmazás inicializálása folyamatban van.</p>
            <p><a href="/debug/system">🔍 System Debug</a> | 
               <a href="/debug/recipes">📊 Recipe Debug</a></p>
            <br>
            <p><small>Ha a probléma továbbra is fennáll, kérjük ellenőrizze a logokat.</small></p>
        </body>
        </html>
        """
    
    @app.route('/debug/system')
    def debug_system_fallback():
        return """
        <h2>Fallback System Debug</h2>
        <p>Az alkalmazás fallback módban fut.</p>
        <p>Valószínű problémák:</p>
        <ul>
            <li>Hiányzó CSV fájlok</li>
            <li>Import hibák</li>
            <li>Database inicializálási problémák</li>
        </ul>
        """
    
    @app.route('/debug/recipes')  
    def debug_recipes_fallback():
        return """
        <h2>Fallback Recipe Debug</h2>
        <p>Receptek nem elérhetők fallback módban.</p>
        <p>Ellenőrizendő:</p>
        <ul>
            <li>hungarian_recipes_github.csv feltöltve?</li>
            <li>recipe_preprocessor.py létezik?</li>
            <li>setup_database.py lefutott?</li>
        </ul>
        """
    
    @app.route('/health')
    def health_fallback():
        return jsonify({
            "status": "fallback",
            "service": "sustainable-recipe-recommender", 
            "version": "2.0-fallback"
        })
    
    return app

# App inicializálás
try:
    app = create_app()
    print("✅ App successfully created")
except Exception as e:
    print(f"❌ App creation failed: {e}")
    app = create_fallback_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
