# export_greenrec_data.py
"""
GreenRec Adatok Export Script
============================
Heroku PostgreSQL adatbázisból való adatok exportálása különböző formátumokban:
- CSV (Excel-kompatibilis)
- JSON (további elemzéshez)
- Summary report (államvizsga bemutatáshoz)
"""

import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import numpy as np

class GreenRecDataExporter:
    def __init__(self, database_url=None):
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        if self.database_url and self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
    
    def get_connection(self):
        """PostgreSQL kapcsolat létrehozása"""
        try:
            return psycopg2.connect(self.database_url, sslmode='require')
        except Exception as e:
            print(f"❌ Adatbázis kapcsolat hiba: {e}")
            return None
    
    def export_all_data(self, output_dir='exports'):
        """Teljes adatok exportálása minden formátumban"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("🚀 GreenRec adatok exportálása...")
        
        # 1. Raw adatok exportálása
        self.export_raw_tables_csv(output_dir, timestamp)
        
        # 2. Összesített analytics exportálása
        self.export_analytics_summary(output_dir, timestamp)
        
        # 3. A/B/C teszt eredmények exportálása
        self.export_ab_test_results(output_dir, timestamp)
        
        # 4. Tanulási görbék exportálása
        self.export_learning_curves(output_dir, timestamp)
        
        # 5. Excel-kompatibilis összefoglaló
        self.export_excel_summary(output_dir, timestamp)
        
        print(f"✅ Export befejezve! Fájlok: {output_dir}/")
        return output_dir

    def export_raw_tables_csv(self, output_dir, timestamp):
        """Raw PostgreSQL táblák exportálása CSV-be"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            tables = {
                'user_sessions': 'SELECT * FROM user_sessions ORDER BY start_time',
                'recipe_ratings': 'SELECT * FROM recipe_ratings ORDER BY user_id, learning_round, timestamp',
                'analytics_metrics': 'SELECT * FROM analytics_metrics ORDER BY user_group, learning_round, timestamp'
            }
            
            for table_name, query in tables.items():
                df = pd.read_sql(query, conn)
                filename = f"{output_dir}/raw_{table_name}_{timestamp}.csv"
                df.to_csv(filename, index=False, encoding='utf-8-sig')  # Excel-kompatibilis
                print(f"📊 Exportálva: {filename} ({len(df)} sor)")
                
        except Exception as e:
            print(f"❌ Raw export hiba: {e}")
        finally:
            conn.close()

    def export_analytics_summary(self, output_dir, timestamp):
        """Analytics összefoglaló exportálása"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            # Csoportonkénti összesítő statisztikák
            query = """
            SELECT 
                user_group,
                COUNT(*) as total_measurements,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(precision_at_5) as avg_precision,
                AVG(recall_at_5) as avg_recall,
                AVG(f1_at_5) as avg_f1,
                AVG(avg_rating) as avg_user_rating,
                STDDEV(f1_at_5) as f1_stddev
            FROM analytics_metrics 
            GROUP BY user_group 
            ORDER BY user_group
            """
            
            df = pd.read_sql(query, conn)
            filename = f"{output_dir}/analytics_summary_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"📈 Exportálva: {filename}")
            
            # JSON verzió is
            summary_dict = df.to_dict('records')
            json_filename = f"{output_dir}/analytics_summary_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, indent=2, ensure_ascii=False)
            print(f"📋 Exportálva: {json_filename}")
            
        except Exception as e:
            print(f"❌ Analytics summary hiba: {e}")
        finally:
            conn.close()

    def export_ab_test_results(self, output_dir, timestamp):
        """A/B/C teszt statisztikai eredmények exportálása"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            # Csoportonkénti teljesítmény összehasonlítás
            query = """
            SELECT 
                user_group,
                learning_round,
                AVG(f1_at_5) as avg_f1_score,
                AVG(precision_at_5) as avg_precision,
                AVG(recall_at_5) as avg_recall,
                COUNT(*) as measurements
            FROM analytics_metrics 
            GROUP BY user_group, learning_round 
            ORDER BY user_group, learning_round
            """
            
            df = pd.read_sql(query, conn)
            
            # Pivot table készítése (csoport vs kör)
            pivot_f1 = df.pivot(index='learning_round', columns='user_group', values='avg_f1_score')
            pivot_precision = df.pivot(index='learning_round', columns='user_group', values='avg_precision')
            
            # Export
            filename = f"{output_dir}/ab_test_results_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            pivot_filename = f"{output_dir}/ab_test_pivot_f1_{timestamp}.csv"
            pivot_f1.to_csv(pivot_filename, encoding='utf-8-sig')
            
            print(f"🧪 A/B/C teszt exportálva: {filename}")
            print(f"📊 Pivot tábla exportálva: {pivot_filename}")
            
        except Exception as e:
            print(f"❌ A/B teszt export hiba: {e}")
        finally:
            conn.close()

    def export_learning_curves(self, output_dir, timestamp):
        """Tanulási görbék adatainak exportálása"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            # Körönkénti fejlődés csoportonként
            query = """
            SELECT 
                user_id,
                user_group,
                learning_round,
                f1_at_5,
                precision_at_5,
                recall_at_5,
                avg_rating,
                timestamp
            FROM analytics_metrics 
            ORDER BY user_group, user_id, learning_round
            """
            
            df = pd.read_sql(query, conn)
            
            # Tanulási görbék számítása
            learning_curves = []
            for group in ['A', 'B', 'C']:
                group_data = df[df['user_group'] == group]
                for round_num in range(1, 6):  # 1-5 körök
                    round_data = group_data[group_data['learning_round'] == round_num]
                    if len(round_data) > 0:
                        learning_curves.append({
                            'group': group,
                            'round': round_num,
                            'avg_f1': round_data['f1_at_5'].mean(),
                            'avg_precision': round_data['precision_at_5'].mean(),
                            'avg_recall': round_data['recall_at_5'].mean(),
                            'user_count': len(round_data),
                            'f1_std': round_data['f1_at_5'].std()
                        })
            
            curves_df = pd.DataFrame(learning_curves)
            
            filename = f"{output_dir}/learning_curves_{timestamp}.csv"
            curves_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"📈 Tanulási görbék exportálva: {filename}")
            
            # Chart.js kompatibilis JSON
            chart_data = {
                'labels': ['1. Kör', '2. Kör', '3. Kör', '4. Kör', '5. Kör'],
                'datasets': []
            }
            
            colors = {'A': '#e74c3c', 'B': '#f39c12', 'C': '#27ae60'}
            for group in ['A', 'B', 'C']:
                group_curves = curves_df[curves_df['group'] == group].sort_values('round')
                if len(group_curves) > 0:
                    chart_data['datasets'].append({
                        'label': f'Csoport {group}',
                        'data': group_curves['avg_f1'].tolist(),
                        'borderColor': colors[group],
                        'backgroundColor': colors[group] + '20'
                    })
            
            chart_filename = f"{output_dir}/learning_curves_chartjs_{timestamp}.json"
            with open(chart_filename, 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Chart.js adatok exportálva: {chart_filename}")
            
        except Exception as e:
            print(f"❌ Learning curves hiba: {e}")
        finally:
            conn.close()

    def export_excel_summary(self, output_dir, timestamp):
        """Excel-kompatibilis összefoglaló exportálása"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            # Államvizsga bemutatáshoz optimalizált összefoglaló
            summary_data = []
            
            # Csoportonkénti végső eredmények
            query = """
            SELECT 
                user_group,
                COUNT(DISTINCT user_id) as felhasznalok_szama,
                AVG(f1_at_5) as atlag_f1_score,
                AVG(precision_at_5) as atlag_precision,
                AVG(recall_at_5) as atlag_recall,
                AVG(avg_rating) as atlag_ertekeles,
                MAX(learning_round) as max_kor
            FROM analytics_metrics 
            GROUP BY user_group 
            ORDER BY user_group
            """
            
            df = pd.read_sql(query, conn)
            
            # Algoritmus nevek hozzáadása
            df['algoritmus'] = df['user_group'].map({
                'A': 'Content-based filtering',
                'B': 'Score-enhanced recommendations', 
                'C': 'Hybrid + XAI approach'
            })
            
            # Relatív teljesítmény számítása (A csoporthoz képest)
            baseline_f1 = df[df['user_group'] == 'A']['atlag_f1_score'].iloc[0] if len(df[df['user_group'] == 'A']) > 0 else 0
            df['relativ_teljesitmeny'] = ((df['atlag_f1_score'] - baseline_f1) / baseline_f1 * 100).round(1)
            
            # Oszlopok átrendezése és formázása
            df_formatted = df[[
                'user_group', 'algoritmus', 'felhasznalok_szama',
                'atlag_f1_score', 'atlag_precision', 'atlag_recall', 
                'atlag_ertekeles', 'relativ_teljesitmeny'
            ]].round(3)
            
            filename = f"{output_dir}/allam_vizsga_osszefoglalo_{timestamp}.csv"
            df_formatted.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"🎓 Államvizsga összefoglaló exportálva: {filename}")
            
            # Statisztikai szignifikancia számítása (egyszerűsített)
            if len(df) >= 2:
                best_group = df.loc[df['atlag_f1_score'].idxmax(), 'user_group']
                best_f1 = df.loc[df['atlag_f1_score'].idxmax(), 'atlag_f1_score']
                improvement = df.loc[df['atlag_f1_score'].idxmax(), 'relativ_teljesitmeny']
                
                print(f"🏆 Legjobb csoport: {best_group} (F1: {best_f1:.3f}, +{improvement:.1f}%)")
            
        except Exception as e:
            print(f"❌ Excel summary hiba: {e}")
        finally:
            conn.close()

    def generate_summary_report(self, output_dir, timestamp):
        """Szöveges összefoglaló jelentés generálása"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("GreenRec A/B/C Teszt Eredmények Összefoglalója")
            report_lines.append("=" * 60)
            report_lines.append(f"Export időpontja: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Alapstatisztikák
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("SELECT COUNT(DISTINCT user_id) as total_users FROM user_sessions")
            total_users = cur.fetchone()['total_users']
            
            cur.execute("SELECT COUNT(*) as total_ratings FROM recipe_ratings")
            total_ratings = cur.fetchone()['total_ratings']
            
            report_lines.append(f"📊 Összesen {total_users} felhasználó vett részt a tesztben")
            report_lines.append(f"⭐ Összesen {total_ratings} recept értékelést gyűjtöttünk")
            report_lines.append("")
            
            # Csoportonkénti eredmények
            cur.execute("""
                SELECT 
                    user_group,
                    COUNT(DISTINCT user_id) as users,
                    AVG(f1_at_5) as avg_f1,
                    AVG(precision_at_5) as avg_precision,
                    AVG(recall_at_5) as avg_recall
                FROM analytics_metrics 
                GROUP BY user_group 
                ORDER BY user_group
            """)
            
            group_results = cur.fetchall()
            algorithms = {
                'A': 'Content-based filtering',
                'B': 'Score-enhanced recommendations',
                'C': 'Hybrid + XAI approach'
            }
            
            report_lines.append("🎯 Csoportonkénti Eredmények:")
            report_lines.append("-" * 40)
            
            baseline_f1 = None
            for result in group_results:
                group = result['user_group']
                f1 = result['avg_f1']
                precision = result['avg_precision'] 
                recall = result['avg_recall']
                users = result['users']
                
                if group == 'A':
                    baseline_f1 = f1
                
                improvement = ""
                if baseline_f1 and group != 'A':
                    improvement_pct = ((f1 - baseline_f1) / baseline_f1 * 100)
                    improvement = f" (+{improvement_pct:.1f}%)"
                
                report_lines.append(f"Csoport {group}: {algorithms[group]}")
                report_lines.append(f"  👥 Felhasználók: {users}")
                report_lines.append(f"  📈 F1@5 Score: {f1:.3f}{improvement}")
                report_lines.append(f"  🎯 Precision@5: {precision:.3f}")
                report_lines.append(f"  🔍 Recall@5: {recall:.3f}")
                report_lines.append("")
            
            cur.close()
            
            # Következtetések
            report_lines.append("🔬 Következtetések:")
            report_lines.append("-" * 20)
            if group_results:
                best_group = max(group_results, key=lambda x: x['avg_f1'])
                report_lines.append(f"• Legjobb teljesítmény: Csoport {best_group['user_group']} ({algorithms[best_group['user_group']]})")
                report_lines.append(f"• F1@5 Score: {best_group['avg_f1']:.3f}")
                
                if baseline_f1:
                    improvement = ((best_group['avg_f1'] - baseline_f1) / baseline_f1 * 100)
                    report_lines.append(f"• Javulás a baseline-hoz képest: +{improvement:.1f}%")
            
            report_lines.append("")
            report_lines.append("=" * 60)
            
            # Fájlba írás
            report_filename = f"{output_dir}/summary_report_{timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            # Konzolra is kiírás
            print('\n'.join(report_lines))
            print(f"📄 Összefoglaló jelentés: {report_filename}")
            
        except Exception as e:
            print(f"❌ Summary report hiba: {e}")
        finally:
            conn.close()

def main():
    """Fő export funkció"""
    # Heroku DATABASE_URL használata
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL környezeti változó nincs beállítva!")
        print("💡 Heroku-n: heroku config -a your-app-name")
        return
    
    exporter = GreenRecDataExporter(database_url)
    
    # Teljes export futtatása
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'greenrec_export_{timestamp}'
    
    exporter.export_all_data(output_dir)
    exporter.generate_summary_report(output_dir, timestamp)
    
    print(f"\n🎉 Minden adat exportálva a '{output_dir}' mappába!")
    print(f"📁 Fájlok: analytics, raw data, A/B teszt, tanulási görbék")

if __name__ == "__main__":
    main()
