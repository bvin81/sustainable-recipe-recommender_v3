#!/usr/bin/env python3
"""
Elemzési eszközök a felhasználói tanulmányhoz
Statisztikai kiértékelés és vizualizáció
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sqlite3
from typing import Dict, List, Tuple
import json

class UserStudyAnalyzer:
    """Felhasználói tanulmány elemzési eszközei"""
    
    def __init__(self, db_path: str = "user_study.db"):
        self.db_path = db_path
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Adatok betöltése az adatbázisból"""
        conn = sqlite3.connect(self.db_path)
        
        participants = pd.read_sql_query('SELECT * FROM participants', conn)
        interactions = pd.read_sql_query('SELECT * FROM interactions', conn)
        questionnaire = pd.read_sql_query('SELECT * FROM questionnaire', conn)
        
        conn.close()
        return participants, interactions, questionnaire
    
    def basic_statistics(self) -> Dict:
        """Alapvető statisztikák"""
        participants, interactions, questionnaire = self.load_data()
        
        stats = {
            'total_participants': len(participants),
            'completed_participants': len(participants[participants['is_completed'] == True]),
            'completion_rate': len(participants[participants['is_completed'] == True]) / len(participants) if len(participants) > 0 else 0,
            'version_distribution': participants['version'].value_counts().to_dict(),
            'avg_interactions_per_user': len(interactions) / len(participants) if len(participants) > 0 else 0
        }
        
        return stats
    
    def compare_versions(self) -> Dict:
        """Verziók összehasonlítása"""
        participants, interactions, questionnaire = self.load_data()
        
        # Questionnaire eredmények verzió szerint
        merged = pd.merge(participants, questionnaire, on='user_id', how='inner')
        
        version_comparison = {}
        metrics = ['system_usability', 'recommendation_quality', 'trust_level', 
                  'explanation_clarity', 'overall_satisfaction']
        
        for metric in metrics:
            version_stats = merged.groupby('version')[metric].agg(['mean', 'std', 'count']).round(3)
            version_comparison[metric] = version_stats.to_dict('index')
        
        # Statisztikai tesztek
        statistical_tests = {}
        for metric in metrics:
            v1_data = merged[merged['version'] == 'v1'][metric].dropna()
            v2_data = merged[merged['version'] == 'v2'][metric].dropna()
            v3_data = merged[merged['version'] == 'v3'][metric].dropna()
            
            if len(v1_data) > 1 and len(v2_data) > 1 and len(v3_data) > 1:
                f_stat, p_value = stats.f_oneway(v1_data, v2_data, v3_data)
                statistical_tests[metric] = {
                    'f_statistic': round(f_stat, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05
                }
        
        return {
            'version_comparison': version_comparison,
            'statistical_tests': statistical_tests
        }
    
    def generate_plots(self):
        """Vizualizációk generálása"""
        participants, interactions, questionnaire = self.load_data()
        
        if len(questionnaire) == 0:
            print("Nincs elegendő adat a vizualizációhoz")
            return
        
        # Merged dataset
        merged = pd.merge(participants, questionnaire, on='user_id', how='inner')
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Study Results - Version Comparison', fontsize=16)
        
        metrics = ['system_usability', 'recommendation_quality', 'trust_level', 
                  'explanation_clarity', 'overall_satisfaction']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            # Box plot verzió szerint
            merged.boxplot(column=metric, by='version', ax=axes[row, col])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Version')
            axes[row, col].set_ylabel('Rating (1-5)')
        
        # Version distribution
        participants['version'].value_counts().plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Version Distribution')
        axes[1, 2].set_xlabel('Version')
        axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plot_path = self.results_dir / "user_study_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_path}")
    
    def export_spss_format(self):
        """SPSS kompatibilis formátum exportálása"""
        participants, interactions, questionnaire = self.load_data()
        
        # Merged dataset SPSS-hez
        merged = pd.merge(participants, questionnaire, on='user_id', how='inner')
        
        # Változók átnevezése SPSS konvenciók szerint
        spss_data = merged.rename(columns={
            'user_id': 'UserID',
            'version': 'Version',
            'age_group': 'AgeGroup', 
            'education': 'Education',
            'cooking_frequency': 'CookingFreq',
            'sustainability_awareness': 'SustainAware',
            'system_usability': 'SystemUsability',
            'recommendation_quality': 'RecommendQuality',
            'trust_level': 'TrustLevel',
            'explanation_clarity': 'ExplanationClarity',
            'overall_satisfaction': 'OverallSatisfaction'
        })
        
        # Kategorikus változók numerikussá alakítása
        version_mapping = {'v1': 1, 'v2': 2, 'v3': 3}
        spss_data['Version_Numeric'] = spss_data['Version'].map(version_mapping)
        
        # Export
        spss_path = self.results_dir / "spss_export"
        spss_path.mkdir(exist_ok=True)
        
       spss_data.to_csv(spss_path / "user_study_data.csv", index=False)
       
       # SPSS syntax fájl generálása
       syntax_content = f"""
* SPSS Syntax for User Study Analysis
* Generated automatically

* Load data
GET DATA
 /TYPE=TXT
 /FILE='user_study_data.csv'
 /ENCODING='UTF8'
 /DELIMITERS=","
 /QUALIFIER='"'
 /ARRANGEMENT=DELIMITED
 /FIRSTCASE=2
 /VARIABLES=
 UserID A12
 Version A2
 AgeGroup A20
 Education A30
 CookingFreq A20
 SustainAware F3.0
 SystemUsability F3.0
 RecommendQuality F3.0
 TrustLevel F3.0
 ExplanationClarity F3.0
 OverallSatisfaction F3.0
 Version_Numeric F1.0.

* Variable labels
VARIABLE LABELS
 UserID 'Unique User Identifier'
 Version 'System Version (v1/v2/v3)'
 Version_Numeric 'System Version (1=v1, 2=v2, 3=v3)'
 SystemUsability 'System Usability Rating'
 RecommendQuality 'Recommendation Quality Rating'
 TrustLevel 'Trust in System Rating'
 ExplanationClarity 'Explanation Clarity Rating'
 OverallSatisfaction 'Overall Satisfaction Rating'.

* Value labels
VALUE LABELS Version_Numeric
 1 'Baseline (v1)'
 2 'Hybrid (v2)'
 3 'Hybrid XAI (v3)'.

* Descriptive statistics
DESCRIPTIVES VARIABLES=SystemUsability RecommendQuality TrustLevel ExplanationClarity OverallSatisfaction
 /STATISTICS=MEAN STDDEV MIN MAX.

* One-way ANOVA for each metric
ONEWAY SystemUsability BY Version_Numeric
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY RecommendQuality BY Version_Numeric
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY TrustLevel BY Version_Numeric
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY ExplanationClarity BY Version_Numeric
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.

ONEWAY OverallSatisfaction BY Version_Numeric
 /STATISTICS DESCRIPTIVES
 /POSTHOC TUKEY.
"""
       
       with open(spss_path / "analysis_syntax.sps", 'w', encoding='utf-8') as f:
           f.write(syntax_content)
       
       print(f"SPSS data exported to: {spss_path}")
   
   def generate_report(self) -> str:
       """HTML riport generálása"""
       basic_stats = self.basic_statistics()
       version_comparison = self.compare_versions()
       
       html_content = f"""
<!DOCTYPE html>
<html>
<head>
   <title>User Study Results Report</title>
   <style>
       body {{ font-family: Arial, sans-serif; margin: 40px; }}
       .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
       .section {{ margin: 30px 0; }}
       .metric {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
       table {{ border-collapse: collapse; width: 100%; }}
       th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
       th {{ background-color: #4CAF50; color: white; }}
       .significant {{ background-color: #ffeb3b; }}
   </style>
</head>
<body>
   <div class="header">
       <h1>🌱 Sustainable Recipe Recommender - User Study Results</h1>
       <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
   </div>
   
   <div class="section">
       <h2>📊 Basic Statistics</h2>
       <div class="metric">
           <strong>Total Participants:</strong> {basic_stats['total_participants']}<br>
           <strong>Completed:</strong> {basic_stats['completed_participants']}<br>
           <strong>Completion Rate:</strong> {basic_stats['completion_rate']:.2%}<br>
       </div>
       
       <h3>Version Distribution</h3>
       <table>
           <tr><th>Version</th><th>Count</th><th>Percentage</th></tr>
"""
       
       total = sum(basic_stats['version_distribution'].values())
       for version, count in basic_stats['version_distribution'].items():
           percentage = count / total * 100 if total > 0 else 0
           html_content += f"<tr><td>{version}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
       
       html_content += """
       </table>
   </div>
   
   <div class="section">
       <h2>🔍 Version Comparison</h2>
"""
       
       for metric, data in version_comparison['version_comparison'].items():
           html_content += f"""
       <div class="metric">
           <h3>{metric.replace('_', ' ').title()}</h3>
           <table>
               <tr><th>Version</th><th>Mean</th><th>Std Dev</th><th>Count</th></tr>
"""
           for version, stats in data.items():
               html_content += f"""
               <tr><td>{version}</td><td>{stats['mean']:.2f}</td><td>{stats['std']:.2f}</td><td>{stats['count']}</td></tr>
"""
           html_content += "</table>"
           
           # Statisztikai teszt eredmény
           if metric in version_comparison['statistical_tests']:
               test_result = version_comparison['statistical_tests'][metric]
               significance = "significant" if test_result['significant'] else ""
               html_content += f"""
           <p class="{significance}">
               <strong>Statistical Test:</strong> F = {test_result['f_statistic']}, p = {test_result['p_value']}
               {' (Significant!)' if test_result['significant'] else ' (Not significant)'}
           </p>
"""
           html_content += "</div>"
       
       html_content += """
   </div>
   
   <div class="section">
       <h2>📈 Key Findings</h2>
       <ul>
"""
       
       # Automatikus insights generálása
       if 'trust_level' in version_comparison['version_comparison']:
           trust_data = version_comparison['version_comparison']['trust_level']
           best_version = max(trust_data.keys(), key=lambda x: trust_data[x]['mean'])
           html_content += f"<li>Highest trust level: <strong>{best_version}</strong> (Mean: {trust_data[best_version]['mean']:.2f})</li>"
       
       if 'overall_satisfaction' in version_comparison['version_comparison']:
           satisfaction_data = version_comparison['version_comparison']['overall_satisfaction']
           best_version = max(satisfaction_data.keys(), key=lambda x: satisfaction_data[x]['mean'])
           html_content += f"<li>Highest satisfaction: <strong>{best_version}</strong> (Mean: {satisfaction_data[best_version]['mean']:.2f})</li>"
       
       # Szignifikáns különbségek
       significant_metrics = [metric for metric, test in version_comparison['statistical_tests'].items() 
                            if test['significant']]
       if significant_metrics:
           html_content += f"<li>Statistically significant differences found in: <strong>{', '.join(significant_metrics)}</strong></li>"
       
       html_content += """
       </ul>
   </div>
   
   <div class="section">
       <h2>💡 Recommendations</h2>
       <ul>
"""
       
       # Automatikus ajánlások
       if len(significant_metrics) > 0:
           html_content += f"<li>Focus on metrics with significant differences: {', '.join(significant_metrics)}</li>"
       
       if basic_stats['completion_rate'] < 0.8:
           html_content += f"<li>Consider improving user experience to increase completion rate (currently {basic_stats['completion_rate']:.1%})</li>"
       
       html_content += """
           <li>Continue data collection for stronger statistical power</li>
           <li>Analyze qualitative feedback from comments</li>
           <li>Consider A/B testing specific features that showed differences</li>
       </ul>
   </div>
   
   <footer style="margin-top: 50px; text-align: center; color: #666;">
       <p>Generated by Sustainable Recipe Recommender Analysis Tools</p>
   </footer>
</body>
</html>
"""
       
       # Riport mentése
       report_path = self.results_dir / "user_study_report.html"
       with open(report_path, 'w', encoding='utf-8') as f:
           f.write(html_content)
       
       print(f"Report generated: {report_path}")
       return str(report_path)

def main():
   """Fő elemzési script"""
   analyzer = UserStudyAnalyzer()
   
   print("🔍 User Study Analysis Starting...")
   print("=" * 50)
   
   # Alapstatisztikák
   basic_stats = analyzer.basic_statistics()
   print(f"📊 Total participants: {basic_stats['total_participants']}")
   print(f"✅ Completed: {basic_stats['completed_participants']}")
   print(f"📈 Completion rate: {basic_stats['completion_rate']:.2%}")
   
   # Verziók összehasonlítása
   if basic_stats['completed_participants'] > 0:
       print("\n🔍 Comparing versions...")
       version_comparison = analyzer.compare_versions()
       
       # Szignifikáns eredmények kiírása
       significant_metrics = [metric for metric, test in version_comparison['statistical_tests'].items() 
                            if test['significant']]
       
       if significant_metrics:
           print(f"🎯 Significant differences found: {', '.join(significant_metrics)}")
       else:
           print("📊 No statistically significant differences yet")
       
       # Vizualizációk
       print("\n📈 Generating plots...")
       analyzer.generate_plots()
       
       # SPSS export
       print("\n📋 Exporting SPSS data...")
       analyzer.export_spss_format()
       
       # HTML riport
       print("\n📄 Generating HTML report...")
       report_path = analyzer.generate_report()
       
       print("=" * 50)
       print("🎉 Analysis complete!")
       print(f"📊 View results: {report_path}")
   
   else:
       print("⚠️ No completed participants yet. Continue data collection.")

if __name__ == "__main__":
   main()
