"""
quick_status.py - בדיקת מצב הפרויקט (מתוקן)
תקן את בעיית ה-Unicode
"""

import os
import sys
import hashlib
from datetime import datetime

def get_file_info(filepath):
    """Get file information safely"""
    try:
        size = os.path.getsize(filepath)
        modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        # Count lines with proper encoding
        lines = 0
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
        except:
            with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
                lines = len(f.readlines())
        
        return size, lines, modified.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        return 0, 0, 'Error'

print("="*70)
print("PROJECT STATUS CHECK - TRADING SYSTEM")
print("="*70)

print("\n1. SYSTEM INFO:")
print("-" * 40)
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")

print("\n2. PROJECT STRUCTURE:")
print("-" * 40)

# מציאת כל קבצי Python
py_files = []
for root, dirs, files in os.walk('.'):
    # דלג על תיקיות לא רלוונטיות
    dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', '.vscode']]
    
    level = root.replace('.', '', 1).count(os.sep)
    indent = ' ' * 2 * level
    
    if level == 0:
        print(f'{os.path.basename(os.getcwd())}/')
    else:
        print(f'{indent}{os.path.basename(root)}/')
    
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            size, lines, modified = get_file_info(filepath)
            py_files.append(filepath)
            print(f'{subindent}{file} ({size:,} bytes, {lines} lines)')

print(f"\nTotal Python files: {len(py_files)}")

print("\n3. KEY FILES STATUS:")
print("-" * 40)

key_files = {
    'main.py': ['scanner_menu', 'run_daily_scan_optimized', 'load_optimized_configuration', 
                'backtest_menu', 'compare_strategies', 'execute_backtest', 'optimizer_menu'],
    'database.py': ['DatabaseManager', 'get_connection', 'initialize'],
    'config.py': ['Config', 'TradingRules', 'get_default_configuration'],
}

for filename, expected_functions in key_files.items():
    if os.path.exists(filename):
        print(f"\n✅ {filename} EXISTS")
        size, lines, modified = get_file_info(filename)
        print(f"   Size: {size:,} bytes | Lines: {lines} | Modified: {modified}")
        
        # בדוק אילו פונקציות קיימות
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            print(f"   Functions found:")
            for func in expected_functions:
                if f'def {func}' in content or f'class {func}' in content:
                    print(f"     ✅ {func}")
                else:
                    print(f"     ❌ {func} - MISSING")
        except Exception as e:
            print(f"   ⚠️ Could not check functions: {e}")
    else:
        print(f"\n❌ {filename} - NOT FOUND")

print("\n4. SCANNER DIRECTORY STATUS:")
print("-" * 40)

scanner_dir = 'scanner'
if os.path.exists(scanner_dir):
    print(f"✅ {scanner_dir}/ directory exists")
    for file in os.listdir(scanner_dir):
        if file.endswith('.py'):
            filepath = os.path.join(scanner_dir, file)
            size, lines, modified = get_file_info(filepath)
            print(f"   • {file}: {lines} lines, {size:,} bytes")
            
            # Check for optimized functions
            if file == 'daily_scanner.py':
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    optimizer_functions = [
                        'load_optimized_configuration',
                        'scan_with_config', 
                        'calculate_weighted_score',
                        'analyze_stock_optimized'
                    ]
                    
                    print("     Optimizer integration:")
                    for func in optimizer_functions:
                        if f'def {func}' in content:
                            print(f"       ✅ {func}")
                        else:
                            print(f"       ❌ {func} - MISSING")
                except:
                    pass
else:
    print(f"❌ {scanner_dir}/ directory NOT FOUND")

print("\n5. OPTIMIZER DIRECTORY STATUS:")
print("-" * 40)

optimizer_dir = 'optimizer'
if os.path.exists(optimizer_dir):
    print(f"✅ {optimizer_dir}/ directory exists")
    for file in os.listdir(optimizer_dir):
        if file.endswith('.py'):
            filepath = os.path.join(optimizer_dir, file)
            size, lines, modified = get_file_info(filepath)
            print(f"   • {file}: {lines} lines, {size:,} bytes")
else:
    print(f"❌ {optimizer_dir}/ directory NOT FOUND")

print("\n6. DATABASE STATUS:")
print("-" * 40)

db_file = 'trading_system.db'
if os.path.exists(db_file):
    size = os.path.getsize(db_file)
    modified = datetime.fromtimestamp(os.path.getmtime(db_file))
    print(f"✅ {db_file}: {size:,} bytes, Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
    
    # Check tables
    try:
        import sqlite3
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"   Tables found: {len(tables)}")
        
        important_tables = ['stocks', 'price_data', 'portfolios', 'trading_configs', 
                          'scan_results', 'backtest_results', 'signals']
        
        for table_name in important_tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone():
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"     ✅ {table_name}: {count} records")
            else:
                print(f"     ❌ {table_name}: NOT FOUND")
        
        # Check for optimized config
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_configs'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM trading_configs WHERE is_active = 1")
            active_configs = cursor.fetchone()[0]
            if active_configs > 0:
                print(f"\n   🎯 OPTIMIZER STATUS: {active_configs} active configuration(s)")
            else:
                print(f"\n   ⚠️ OPTIMIZER STATUS: No active configuration")
        
        conn.close()
    except Exception as e:
        print(f"   ⚠️ Could not check database: {e}")
else:
    print(f"❌ {db_file} - NOT FOUND")

print("\n7. RECENT ERRORS CHECK:")
print("-" * 40)

# Check for log files
log_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.log'):
            log_files.append(os.path.join(root, file))

if log_files:
    print(f"Found {len(log_files)} log files:")
    for log_file in log_files:
        size = os.path.getsize(log_file)
        print(f"   • {log_file}: {size:,} bytes")
else:
    print("No log files found")

print("\n8. CONFIGURATION SUMMARY:")
print("-" * 40)

# Check what features are integrated
features = {
    'Scanner with Optimizer': False,
    'Backtest Module': False,
    'Portfolio Management': False,
    'Genetic Algorithm': False,
    'Database Manager': False
}

if os.path.exists('main.py'):
    try:
        with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
        
        if 'load_optimized_configuration' in main_content:
            features['Scanner with Optimizer'] = True
        if 'backtest_menu' in main_content and 'execute_backtest' in main_content:
            features['Backtest Module'] = True
        if 'portfolio_menu' in main_content:
            features['Portfolio Management'] = True
        if 'GeneticOptimizer' in main_content or 'genetic_algorithm' in main_content:
            features['Genetic Algorithm'] = True
        if 'DatabaseManager' in main_content:
            features['Database Manager'] = True
    except:
        pass

print("Feature Integration Status:")
for feature, status in features.items():
    status_icon = "✅" if status else "❌"
    print(f"   {status_icon} {feature}")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

total_py_lines = 0
for py_file in py_files:
    try:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            total_py_lines += len(f.readlines())
    except:
        pass

print(f"• Total Python files: {len(py_files)}")
print(f"• Total lines of code: {total_py_lines:,}")
print(f"• Project size: {sum(os.path.getsize(f) for f in py_files):,} bytes")

# Check critical issues
critical_issues = []

if not os.path.exists('main.py'):
    critical_issues.append("main.py missing")
if not os.path.exists('trading_system.db'):
    critical_issues.append("Database not found")
if not features['Scanner with Optimizer']:
    critical_issues.append("Scanner not integrated with optimizer")
if not features['Backtest Module']:
    critical_issues.append("Backtest module not integrated")

if critical_issues:
    print(f"\n⚠️ CRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print(f"   • {issue}")
else:
    print(f"\n✅ No critical issues found!")

print("\n" + "="*70)
print("Report generated successfully!")
print("="*70)

# Save report to file
try:
    with open('project_status_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"PROJECT STATUS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Total Files: {len(py_files)}\n")
        f.write(f"Total Lines: {total_py_lines}\n")
        f.write(f"\nFeatures Status:\n")
        for feature, status in features.items():
            f.write(f"  {'✓' if status else '✗'} {feature}\n")
        if critical_issues:
            f.write(f"\nCritical Issues:\n")
            for issue in critical_issues:
                f.write(f"  • {issue}\n")
    
    print(f"\n📄 Full report saved to: project_status_report.txt")
except Exception as e:
    print(f"Could not save report: {e}")