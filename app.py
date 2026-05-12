from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
CSV_PATH   = os.path.join(BASE_DIR, 'Personalized_Diet_Recommendations.csv')
DB_PATH    = os.path.join(BASE_DIR, 'database.db')

# ─── Load Model & CSV ─────────────────────────────────────────────────────────
model   = joblib.load(MODEL_PATH)
diet_df = pd.read_csv(CSV_PATH, encoding='latin-1')
print(f"✅ Model loaded | ✅ Diet CSV: {len(diet_df)} records")

# ─── Database Setup ───────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT    UNIQUE NOT NULL,
        email        TEXT    UNIQUE NOT NULL,
        password     TEXT    NOT NULL,
        created_at   TEXT    NOT NULL,
        last_login   TEXT
    )''')

    # Login sessions log
    c.execute('''CREATE TABLE IF NOT EXISTS login_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        login_at   TEXT    NOT NULL,
        logout_at  TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # Predictions history
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL,
        pregnancies   REAL,
        glucose       REAL,
        blood_pressure REAL,
        skin_thickness REAL,
        insulin       REAL,
        bmi           REAL,
        dpf           REAL,
        age           REAL,
        result        TEXT    NOT NULL,
        probability   REAL    NOT NULL,
        risk_level    TEXT    NOT NULL,
        bmi_category  TEXT,
        glucose_status TEXT,
        created_at    TEXT    NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # Survey table
    c.execute('''CREATE TABLE IF NOT EXISTS surveys (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL,
        prediction_id INTEGER NOT NULL,
        q1  INTEGER NOT NULL, q2  INTEGER NOT NULL, q3  INTEGER NOT NULL,
        q4  INTEGER NOT NULL, q5  INTEGER NOT NULL, q6  INTEGER NOT NULL,
        q7  INTEGER NOT NULL, q8  INTEGER NOT NULL, q9  INTEGER NOT NULL,
        q10 INTEGER NOT NULL,
        avg_score     REAL NOT NULL,
        submitted_at  TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(prediction_id) REFERENCES predictions(id),
        UNIQUE(user_id, prediction_id)
    )''')

    # Add survey_done column to users if not exists
    try:
        c.execute('ALTER TABLE users ADD COLUMN survey_done INTEGER DEFAULT 0')
    except Exception:
        pass  # column already exists
    try:
        c.execute('ALTER TABLE users ADD COLUMN first_survey_reminder_shown INTEGER DEFAULT 0')
    except Exception:
        pass  # column already exists

    # Migrate old surveys schema (one survey per user) to per-analysis surveys
    col_rows = c.execute("PRAGMA table_info(surveys)").fetchall()
    survey_cols = {row[1] for row in col_rows}
    needs_migration = 'prediction_id' not in survey_cols

    if not needs_migration:
        idx_rows = c.execute("PRAGMA index_list('surveys')").fetchall()
        for idx in idx_rows:
            # pragma index_list: (seq, name, unique, origin, partial)
            idx_name = idx[1]
            is_unique = bool(idx[2])
            if not is_unique:
                continue
            idx_info = c.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
            idx_cols = [r[2] for r in idx_info]
            if idx_cols == ['user_id']:
                needs_migration = True
                break

    if needs_migration:
        c.execute('''CREATE TABLE IF NOT EXISTS surveys_new (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            q1  INTEGER NOT NULL, q2  INTEGER NOT NULL, q3  INTEGER NOT NULL,
            q4  INTEGER NOT NULL, q5  INTEGER NOT NULL, q6  INTEGER NOT NULL,
            q7  INTEGER NOT NULL, q8  INTEGER NOT NULL, q9  INTEGER NOT NULL,
            q10 INTEGER NOT NULL,
            avg_score     REAL NOT NULL,
            submitted_at  TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(prediction_id) REFERENCES predictions(id),
            UNIQUE(user_id, prediction_id)
        )''')

        if 'prediction_id' in survey_cols:
            c.execute('''INSERT OR IGNORE INTO surveys_new
                (user_id,prediction_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,avg_score,submitted_at)
                SELECT s.user_id,
                       COALESCE(s.prediction_id,
                           (SELECT p.id FROM predictions p
                            WHERE p.user_id = s.user_id
                            ORDER BY p.created_at DESC, p.id DESC LIMIT 1)),
                       s.q1,s.q2,s.q3,s.q4,s.q5,s.q6,s.q7,s.q8,s.q9,s.q10,s.avg_score,s.submitted_at
                FROM surveys s
                WHERE COALESCE(s.prediction_id,
                    (SELECT p.id FROM predictions p
                     WHERE p.user_id = s.user_id
                     ORDER BY p.created_at DESC, p.id DESC LIMIT 1)) IS NOT NULL
            ''')
        else:
            c.execute('''INSERT OR IGNORE INTO surveys_new
                (user_id,prediction_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,avg_score,submitted_at)
                SELECT s.user_id,
                       (SELECT p.id FROM predictions p
                        WHERE p.user_id = s.user_id
                        ORDER BY p.created_at DESC, p.id DESC LIMIT 1),
                       s.q1,s.q2,s.q3,s.q4,s.q5,s.q6,s.q7,s.q8,s.q9,s.q10,s.avg_score,s.submitted_at
                FROM surveys s
                WHERE (SELECT p.id FROM predictions p
                       WHERE p.user_id = s.user_id
                       ORDER BY p.created_at DESC, p.id DESC LIMIT 1) IS NOT NULL
            ''')

        c.execute('DROP TABLE surveys')
        c.execute('ALTER TABLE surveys_new RENAME TO surveys')

    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_surveys_user_prediction ON surveys(user_id, prediction_id)')

    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_db()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def current_user():
    if 'user_id' not in session:
        return None
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    conn.close()
    return user

def login_required_json():
    """Return error JSON if not logged in."""
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Login required', 'redirect': '/login'}), 401
    return None

# ─── Feature Engineering ──────────────────────────────────────────────────────
def build_features(data):
    preg = float(data.get('pregnancies', 0))
    gluc = float(data.get('glucose', 0))
    bp   = float(data.get('bloodPressure', 0))
    skin = float(data.get('skinThickness', 0))
    ins  = float(data.get('insulin', 0))
    bmi  = float(data.get('bmi', 0))
    dpf  = float(data.get('dpf', 0))
    age  = float(data.get('age', 0))
    N1=int(age<=30 and gluc<=120); N2=int(bmi<=30)
    N4=int(gluc<=105 and bp<=80);  N7=int(gluc<=105 and bmi<=30)
    N0=bmi*skin; N8=preg/(age+1); N12=age*dpf; N13=gluc/(dpf+0.001)
    return pd.DataFrame([{
        'Pregnancies':preg,'Glucose':gluc,'BloodPressure':bp,
        'SkinThickness':skin,'Insulin':ins,'BMI':bmi,
        'DiabetesPedigreeFunction':dpf,'Age':age,
        'N1':N1,'N2':N2,'N4':N4,'N7':N7,'N0':N0,'N8':N8,'N12':N12,'N13':N13}])

def get_diet_base(age, bmi, is_diabetic):
    df = diet_df.copy()
    if is_diabetic:
        d = df[df['Chronic_Disease'].str.contains('Diabetes', case=False, na=False)]
        if len(d) >= 5: df = d
    df = df.copy()
    df['score'] = abs(df['Age']-age) + abs(df['BMI']-bmi)*2
    best = df.nsmallest(1,'score').iloc[0]
    return {
        'meal_plan': str(best.get('Recommended_Meal_Plan','Balanced Diet')),
        'calories':  int(best.get('Recommended_Calories',2000)),
        'protein':   int(best.get('Recommended_Protein',80)),
        'carbs':     int(best.get('Recommended_Carbs',250)),
        'fats':      int(best.get('Recommended_Fats',70)),
        'exercise_frequency': int(best.get('Exercise_Frequency',3)),
    }

def build_nutrition(age, bmi, glucose, insulin, is_diabetic, base):
    cal=base['calories']; carbs=base['carbs']; protein=base['protein']; fats=base['fats']
    tips=[]; foods_ok=[]; foods_avoid=[]
    if bmi>=35:   cal=int(cal*0.80); tips.append({'icon':'⚠️','color':'#dc2626','text':'High calorie reduction needed — BMI is very high (Obese class II+)'})
    elif bmi>=30: cal=int(cal*0.88); tips.append({'icon':'⚠️','color':'#ea580c','text':'Moderate calorie reduction — BMI indicates obesity'})
    elif bmi<18.5:cal=int(cal*1.15); tips.append({'icon':'ℹ️','color':'#2563eb','text':'Calorie increase needed — BMI is below normal'})
    else:                             tips.append({'icon':'✅','color':'#059669','text':'Calorie target is ideal for your healthy BMI'})
    if glucose>=200:
        carbs=int(carbs*0.55); tips.append({'icon':'🔴','color':'#dc2626','text':'Very low-carb diet critical — glucose dangerously high (≥200)'})
        foods_avoid+=['White bread & rice','Pasta & noodles','Sweets & desserts','Fruit juices & sodas','Potatoes & starchy foods']
        foods_ok+=['Non-starchy vegetables (broccoli, spinach, kale)','Lean proteins (chicken, fish, eggs)','Avocado & olive oil','Nuts & seeds','Berries only']
    elif glucose>=140:
        carbs=int(carbs*0.70); tips.append({'icon':'🟠','color':'#ea580c','text':'Low-carb diet strongly recommended — glucose above normal'})
        foods_avoid+=['Refined sugars','White rice & bread','Sugary drinks']
        foods_ok+=['Whole grains (oats, quinoa, brown rice)','Green vegetables','Lean proteins','Legumes','Low-sugar fruits']
    elif glucose>=100:
        carbs=int(carbs*0.85); tips.append({'icon':'🟡','color':'#d97706','text':'Moderate carb control — glucose slightly above normal'})
        foods_avoid+=['Refined sugars','Sugary beverages','Processed snacks']
        foods_ok+=['Whole grains','Vegetables & most fruits','Lean proteins']
    else:
        tips.append({'icon':'✅','color':'#059669','text':'Glucose normal — maintain balanced carbohydrate intake'})
        foods_ok+=['Whole grains','All fruits & vegetables','Lean proteins','Healthy fats']
    if age>=60:   protein=int(protein*1.20); tips.append({'icon':'ℹ️','color':'#7c3aed','text':'Higher protein needed — muscle preservation critical at 60+'})
    elif age<=25: protein=int(protein*1.10); tips.append({'icon':'ℹ️','color':'#0d9488','text':'Slightly more protein for muscle growth at your age'})
    if insulin>200: fats=int(fats*0.85); tips.append({'icon':'🟠','color':'#ea580c','text':'Reduce saturated fats — high insulin levels detected'})
    return {'meal_plan':base['meal_plan'],'calories':cal,'protein':protein,'carbs':carbs,'fats':fats,
            'tips':tips,'foods_ok':list(dict.fromkeys(foods_ok)),'foods_avoid':list(dict.fromkeys(foods_avoid))}

def build_exercise(age, bmi, glucose, bp, freq):
    if bmi>=35 or age>=65:   intensity='Low';          mins=20
    elif bmi>=30 or age>=50: intensity='Moderate';     mins=30
    else:                    intensity='Moderate–High'; mins=40
    exercises=[]
    if bmi>=30: exercises.append({'icon':'🚶','name':'Brisk Walking','freq':f'{freq+1}x/week','duration':f'{mins} min','why':'Best for weight loss & blood sugar control with low joint stress'})
    else:       exercises.append({'icon':'🏃','name':'Jogging / Running','freq':f'{freq}x/week','duration':f'{mins} min','why':'Burns calories efficiently and boosts cardiovascular health'})
    if age>=60: exercises.append({'icon':'🏋️','name':'Light Resistance Training','freq':'2x/week','duration':'20 min','why':'Preserves muscle mass and bone density — critical at 60+'})
    else:       exercises.append({'icon':'🏋️','name':'Strength Training','freq':'3x/week','duration':'25–35 min','why':'Builds muscle that burns glucose even at rest'})
    exercises.append({'icon':'🧘','name':'Yoga / Stretching','freq':'Daily','duration':'15 min','why':'Lowers cortisol — stress hormone that raises blood sugar'})
    if bmi>=32: exercises.append({'icon':'🏊','name':'Swimming / Water Aerobics','freq':'2x/week','duration':'30 min','why':'Zero joint pressure — perfect for high BMI'})
    else:       exercises.append({'icon':'🚴','name':'Cycling','freq':'2x/week','duration':'30 min','why':'Great low-impact cardio that burns fat effectively'})
    tips=[]; warnings_list=[]
    if glucose>=140: tips+=['Check blood sugar BEFORE and AFTER every workout','Keep a snack nearby during exercise (hypoglycemia risk)']
    if bp>=90: warnings_list.append('🔴 High BP — always warm up 10 min, avoid heavy lifts')
    if age>=60: tips.append('Start with 10 min/day and build up gradually over weeks')
    if bmi>=30: tips.append('Consistency beats intensity — 5 days x 20 min beats 1 day x 2 hrs')
    tips.append(f'Weekly target: {freq} sessions, {mins} min each — Intensity: {intensity}')
    return {'exercises':exercises,'tips':tips,'warnings':warnings_list,'sessions':freq,'intensity':intensity,'session_min':mins}

def build_monitoring(age, bmi, glucose, bp, insulin, is_diabetic):
    checks=[]; schedule=[]
    if is_diabetic:
        if glucose>=200:
            checks.append({'icon':'🩺','color':'#dc2626','bg':'#fef2f2','text':'Check glucose 4x/day — CRITICALLY HIGH'})
            schedule+=[{'time':'Every morning','task':'Fasting glucose check'},{'time':'After each meal','task':'Post-meal check'},{'time':'Before bed','task':'Bedtime check'}]
        elif glucose>=140:
            checks.append({'icon':'🩺','color':'#ea580c','bg':'#fff7ed','text':'Check glucose 2–3x/day — above normal'})
            schedule+=[{'time':'Every morning','task':'Fasting check'},{'time':'2hrs after lunch','task':'Post-meal check'}]
        else:
            checks.append({'icon':'🩺','color':'#d97706','bg':'#fffbeb','text':'Check glucose once daily (morning) — controlled'})
            schedule.append({'time':'Every morning','task':'Fasting glucose check'})
    else:
        checks.append({'icon':'🩺','color':'#059669','bg':'#ecfdf5','text':'Annual glucose screening sufficient — non-diabetic'})
        schedule.append({'time':'Once a year','task':'Full blood panel'})
    if bp>=90:   checks.append({'icon':'💓','color':'#dc2626','bg':'#fef2f2','text':'Monitor blood pressure DAILY — currently HIGH'})
    elif bp>=80: checks.append({'icon':'💓','color':'#d97706','bg':'#fffbeb','text':'Monitor BP 3x/week — slightly elevated'})
    else:        checks.append({'icon':'💓','color':'#059669','bg':'#ecfdf5','text':'BP normal — monthly check is fine'})
    if bmi>=30:  checks.append({'icon':'⚖️','color':'#ea580c','bg':'#fff7ed','text':'Weigh yourself weekly — track weight loss'})
    else:        checks.append({'icon':'⚖️','color':'#059669','bg':'#ecfdf5','text':'Weigh yourself monthly — maintain healthy weight'})
    if insulin>200: checks.append({'icon':'💉','color':'#7c3aed','bg':'#fdf4ff','text':'Insulin very high — discuss medication with doctor ASAP'})
    if is_diabetic and glucose>=200:
        checks.append({'icon':'🏥','color':'#dc2626','bg':'#fef2f2','text':'Doctor visit every 4 weeks — critically high glucose'})
        schedule.append({'time':'Every 4 weeks','task':'Doctor appointment'})
    elif is_diabetic:
        checks.append({'icon':'🏥','color':'#2563eb','bg':'#eff6ff','text':'Doctor every 3 months — standard diabetic care'})
        schedule.append({'time':'Every 3 months','task':'Doctor + HbA1c test'})
    else:
        checks.append({'icon':'🏥','color':'#059669','bg':'#ecfdf5','text':'Annual full check-up recommended'})
    checks.append({'icon':'😴','color':'#0d9488','bg':'#f0fdfa','text':'Sleep 7–8 hours/night — poor sleep raises blood sugar 23%'})
    schedule.append({'time':'Every night','task':'7–8 hours consistent sleep'})
    return {'checks':checks,'schedule':schedule}

def build_mental(age, bmi, glucose, is_diabetic, prob_pct):
    if is_diabetic and glucose>=200:
        tips=[
            {'icon':'🧘','color':'#7c3aed','bg':'#fdf4ff','title':'Mindful Breathing','desc':'5 min deep breathing 3x/day — lowers cortisol that raises blood sugar'},
            {'icon':'📓','color':'#2563eb','bg':'#eff6ff','title':'Daily Health Journal','desc':'Log glucose, meals, mood daily — patterns give you control'},
            {'icon':'👨‍⚕️','color':'#dc2626','bg':'#fef2f2','title':'Professional Support','desc':'Consider a diabetes counselor — you should not face this alone'},
            {'icon':'🎯','color':'#059669','bg':'#ecfdf5','title':'Small Weekly Goals','desc':'One improvement/week — small wins build big confidence'},
        ]
        quote='"Every day you manage your diabetes is a victory. You are stronger than you think. 💪"'
    elif is_diabetic:
        tips=[
            {'icon':'🧘','color':'#7c3aed','bg':'#fdf4ff','title':'Morning Meditation','desc':'10 min each morning — reduces stress that spikes blood sugar'},
            {'icon':'🌳','color':'#059669','bg':'#ecfdf5','title':'Nature Walks','desc':'15 min outside improves mood and vitamin D levels'},
            {'icon':'👨‍👩‍👧','color':'#2563eb','bg':'#eff6ff','title':'Social Connection','desc':'Share your journey — support system is critical for success'},
            {'icon':'🎵','color':'#ea580c','bg':'#fff7ed','title':'Music Therapy','desc':'Calm music for 20 min during meals lowers cortisol'},
        ]
        quote='"Diabetes is manageable — with the right steps, you can live a full, healthy life. ✨"'
    elif prob_pct>=30:
        tips=[
            {'icon':'⚡','color':'#d97706','bg':'#fffbeb','title':'Stay Proactive','desc':'Moderate risk — now is the best time to build healthy habits'},
            {'icon':'🧘','color':'#7c3aed','bg':'#fdf4ff','title':'Stress Management','desc':'Chronic stress is a major diabetes risk — 10 min daily mindfulness helps'},
            {'icon':'😴','color':'#0d9488','bg':'#f0fdfa','title':'Prioritize Sleep','desc':'Poor sleep increases insulin resistance — 7–8 hrs is as powerful as medication'},
            {'icon':'📱','color':'#2563eb','bg':'#eff6ff','title':'Health Tracking','desc':'Track steps, sleep, meals — awareness leads to better choices'},
        ]
        quote='"Prevention is the most powerful medicine. Every good habit today protects your tomorrow. 🌟"'
    else:
        tips=[
            {'icon':'🌟','color':'#059669','bg':'#ecfdf5','title':'Celebrate Your Health','desc':'You are in great shape! Acknowledge your healthy choices daily'},
            {'icon':'🧘','color':'#7c3aed','bg':'#fdf4ff','title':'Maintain Mindfulness','desc':'5–10 min daily meditation keeps you sharp and calm'},
            {'icon':'👥','color':'#2563eb','bg':'#eff6ff','title':'Inspire Others','desc':'Your healthy lifestyle can motivate the people around you'},
            {'icon':'📈','color':'#d97706','bg':'#fffbeb','title':'Keep Improving','desc':'Set new wellness goals each month — health is a lifelong journey'},
        ]
        quote='"Health is not just what you eat — it is also what you think, say, and do every day. 💚"'
    if age>=60: tips.append({'icon':'🤝','color':'#db2777','bg':'#fdf2f8','title':'Stay Socially Active','desc':'Social connection is critical for health at 60+'})
    elif age<=30: tips.append({'icon':'💪','color':'#059669','bg':'#ecfdf5','title':'Build Habits Now','desc':'Habits built in your 20s–30s define your health at 50+'})
    return {'tips':tips,'quote':quote}

def interpret_risk(prob):
    if prob>=0.75:   return {'level':'High Risk',     'color':'red',    'icon':'exclamation-triangle'}
    elif prob>=0.50: return {'level':'Moderate Risk', 'color':'orange', 'icon':'exclamation-circle'}
    elif prob>=0.30: return {'level':'Low Risk',      'color':'yellow', 'icon':'info-circle'}
    else:            return {'level':'Very Low Risk', 'color':'green',  'icon':'check-circle'}

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    data     = request.get_json()
    username = data.get('username','').strip()
    email    = data.get('email','').strip().lower()
    password = data.get('password','')
    if not username or not email or not password:
        return jsonify({'status':'error','message':'All fields are required'}), 400
    if len(password) < 6:
        return jsonify({'status':'error','message':'Password must be at least 6 characters'}), 400
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO users (username,email,password,created_at) VALUES (?,?,?,?)',
            (username, email, hash_password(password), datetime.now().isoformat()))
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
        session['user_id']   = user['id']
        session['username']  = user['username']
        log_id = conn.execute('INSERT INTO login_logs (user_id,login_at) VALUES (?,?)',
            (user['id'], datetime.now().isoformat())).lastrowid
        session['log_id'] = log_id
        conn.commit()
        return jsonify({'status':'success','message':'Account created!','redirect':'/dashboard'})
    except sqlite3.IntegrityError as e:
        msg = 'Username already taken' if 'username' in str(e) else 'Email already registered'
        return jsonify({'status':'error','message':msg}), 409
    finally:
        conn.close()

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    data     = request.get_json()
    email    = data.get('email','').strip().lower()
    password = data.get('password','')
    if not email or not password:
        return jsonify({'status':'error','message':'Email and password are required'}), 400
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email=? AND password=?',
        (email, hash_password(password))).fetchone()
    if not user:
        conn.close()
        return jsonify({'status':'error','message':'Invalid email or password'}), 401
    session['user_id']  = user['id']
    session['username'] = user['username']
    conn.execute('UPDATE users SET last_login=? WHERE id=?',(datetime.now().isoformat(), user['id']))
    log_id = conn.execute('INSERT INTO login_logs (user_id,login_at) VALUES (?,?)',
        (user['id'], datetime.now().isoformat())).lastrowid
    session['log_id'] = log_id
    conn.commit()
    conn.close()
    return jsonify({'status':'success','message':f'Welcome back, {user["username"]}!','redirect':'/dashboard'})

@app.route('/logout')
def logout():
    if 'user_id' in session:
        conn = get_db()
        latest_pred = conn.execute(
            'SELECT id FROM predictions WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
        can_take_survey = False
        if latest_pred:
            existing = conn.execute(
                'SELECT id FROM surveys WHERE user_id=? AND prediction_id=?',
                (session['user_id'], latest_pred['id'])
            ).fetchone()
            can_take_survey = existing is None

        # If latest analysis has no survey yet → block logout and redirect to survey
        if can_take_survey:
            conn.execute('UPDATE users SET survey_done=0 WHERE id=?', (session['user_id'],))
            conn.commit()
            conn.close()
            return redirect('/survey?required=1')
        conn.execute('UPDATE users SET survey_done=1 WHERE id=?', (session['user_id'],))
        conn.commit()
        if 'log_id' in session:
            conn.execute('UPDATE login_logs SET logout_at=? WHERE id=?',
                (datetime.now().isoformat(), session['log_id']))
            conn.commit()
        conn.close()
    session.clear()
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    user = current_user()
    if not user:
        return redirect('/login')
    conn = get_db()
    preds = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
        (user['id'],)).fetchall()
    latest_prediction_id = preds[0]['id'] if len(preds) > 0 else None
    can_take_survey = False
    if latest_prediction_id:
        latest_survey = conn.execute(
            'SELECT id FROM surveys WHERE user_id=? AND prediction_id=?',
            (user['id'], latest_prediction_id)).fetchone()
        can_take_survey = latest_survey is None

    first_survey_reminder = False
    reminder_already_shown = bool(user['first_survey_reminder_shown']) if 'first_survey_reminder_shown' in user.keys() else False
    if len(preds) == 1 and can_take_survey and not reminder_already_shown:
        first_survey_reminder = True
        conn.execute('UPDATE users SET first_survey_reminder_shown=1 WHERE id=?', (user['id'],))

    total_logins = conn.execute(
        'SELECT COUNT(*) as c FROM login_logs WHERE user_id=?',(user['id'],)).fetchone()['c']
    survey_count = conn.execute(
        'SELECT COUNT(*) as c FROM surveys WHERE user_id=?', (user['id'],)).fetchone()['c']
    if latest_prediction_id:
        conn.execute(
            'UPDATE users SET survey_done=? WHERE id=?',
            (0 if can_take_survey else 1, user['id'])
        )
    conn.commit()
    conn.close()
    predictions = [dict(p) for p in preds]
    return render_template('dashboard.html',
        user=dict(user),
        predictions=predictions,
        total_logins=total_logins,
        can_take_survey=can_take_survey,
        survey_count=survey_count,
        first_survey_reminder=first_survey_reminder)

@app.route('/api/me')
def api_me():
    user = current_user()
    if not user:
        return jsonify({'logged_in': False})
    conn = get_db()
    latest_pred = conn.execute(
        'SELECT id FROM predictions WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1',
        (user['id'],)
    ).fetchone()
    has_pred = latest_pred is not None
    can_take_survey = False
    if latest_pred:
        existing = conn.execute(
            'SELECT id FROM surveys WHERE user_id=? AND prediction_id=?',
            (user['id'], latest_pred['id'])
        ).fetchone()
        can_take_survey = existing is None
    conn.close()
    return jsonify({
        'logged_in':    True,
        'username':     user['username'],
        'email':        user['email'],
        'survey_done':  not can_take_survey if has_pred else False,
        'can_take_survey': can_take_survey,
        'has_prediction': has_pred,
    })

# ─── Main Routes ──────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    err = login_required_json()
    if err: return err
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status':'error','message':'No data received'}), 400
        required = ['pregnancies','glucose','bloodPressure','skinThickness','insulin','bmi','dpf','age']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'status':'error','message':f'Missing: {missing}'}), 400

        age=float(data.get('age',30)); bmi=float(data.get('bmi',0))
        glucose=float(data.get('glucose',0)); bp=float(data.get('bloodPressure',0))
        insulin=float(data.get('insulin',0))

        X             = build_features(data)
        prediction    = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        prob_diabetic = float(probabilities[1])
        is_diabetic   = prediction == 1

        if bmi<18.5:   bmi_cat='Underweight'
        elif bmi<25:   bmi_cat='Normal'
        elif bmi<30:   bmi_cat='Overweight'
        else:          bmi_cat='Obese'

        if glucose<100:   glucose_status='Normal'
        elif glucose<126: glucose_status='Prediabetes'
        else:             glucose_status='Diabetes Range'

        if bp<80:    bp_status='Normal'
        elif bp<90:  bp_status='Elevated'
        else:        bp_status='High'

        risk = interpret_risk(prob_diabetic)

        # Save prediction to DB
        conn = get_db()
        new_prediction_id = conn.execute('''INSERT INTO predictions
            (user_id,pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age,
             result,probability,risk_level,bmi_category,glucose_status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (session['user_id'],
             float(data.get('pregnancies',0)), glucose, bp,
             float(data.get('skinThickness',0)), insulin, bmi,
             float(data.get('dpf',0)), age,
             'DIABETIC' if is_diabetic else 'NON-DIABETIC',
             round(prob_diabetic*100,1), risk['level'], bmi_cat, glucose_status,
             datetime.now().isoformat())).lastrowid

        pred_count = conn.execute(
            'SELECT COUNT(*) as c FROM predictions WHERE user_id=?', (session['user_id'],)
        ).fetchone()['c']
        reminder_row = conn.execute(
            'SELECT first_survey_reminder_shown FROM users WHERE id=?', (session['user_id'],)
        ).fetchone()
        reminder_shown = bool(reminder_row['first_survey_reminder_shown']) if reminder_row and 'first_survey_reminder_shown' in reminder_row.keys() else False
        remind_survey_first_time = pred_count == 1 and not reminder_shown
        if remind_survey_first_time:
            conn.execute('UPDATE users SET first_survey_reminder_shown=1 WHERE id=?', (session['user_id'],))
        conn.commit()
        conn.close()

        base       = get_diet_base(age, bmi, is_diabetic)
        nutrition  = build_nutrition(age, bmi, glucose, insulin, is_diabetic, base)
        exercise   = build_exercise(age, bmi, glucose, bp, base['exercise_frequency'])
        monitoring = build_monitoring(age, bmi, glucose, bp, insulin, is_diabetic)
        mental     = build_mental(age, bmi, glucose, is_diabetic, prob_diabetic*100)

        return jsonify({
            'status':'success',
            'prediction':'DIABETIC' if is_diabetic else 'NON-DIABETIC',
            'is_diabetic':is_diabetic,
            'probability':{'diabetic':round(prob_diabetic*100,1),'healthy':round((1-prob_diabetic)*100,1)},
            'risk':risk,'bmi':round(bmi,1),'bmi_category':bmi_cat,
            'glucose':round(glucose,1),'glucose_status':glucose_status,'bp_status':bp_status,
            'nutrition':nutrition,'exercise':exercise,'monitoring':monitoring,'mental':mental,
            'survey': {
                'can_take_now': True,
                'prediction_id': new_prediction_id,
                'remind_first_time': remind_survey_first_time
            }
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status':'error','message':str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status':'ok','model':type(model).__name__})


# ─── Survey Routes ────────────────────────────────────────────────────────────
@app.route('/survey')
def survey_page():
    user = current_user()
    if not user:
        return redirect('/login')
    conn = get_db()
    # Must have at least 1 prediction to access survey
    latest_pred = conn.execute(
        'SELECT id FROM predictions WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1',
        (user['id'],)
    ).fetchone()
    if not latest_pred:
        conn.close()
        return redirect('/dashboard?survey=nopred')

    # Check if latest analysis already has survey
    existing = conn.execute(
        'SELECT id FROM surveys WHERE user_id=? AND prediction_id=?',
        (user['id'], latest_pred['id'])
    ).fetchone()
    conn.close()
    if existing:
        return redirect('/dashboard?survey=done')
    return render_template('survey.html', user=dict(user),
                           required=request.args.get('required','0'))

@app.route('/survey/submit', methods=['POST'])
def survey_submit():
    user = current_user()
    if not user:
        return jsonify({'status':'error','message':'Login required'}), 401
    data = request.get_json()
    answers = []
    for i in range(1, 11):
        val = data.get(f'q{i}')
        if val is None or int(val) not in [1,2,3,4,5]:
            return jsonify({'status':'error','message':f'Question {i} is missing or invalid'}), 400
        answers.append(int(val))
    avg = round(sum(answers)/len(answers), 2)
    conn = get_db()
    try:
        latest_pred = conn.execute(
            'SELECT id FROM predictions WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1',
            (user['id'],)
        ).fetchone()
        if not latest_pred:
            conn.close()
            return jsonify({'status':'error','message':'Run at least one analysis before filling the survey.'}), 400

        existing = conn.execute(
            'SELECT id FROM surveys WHERE user_id=? AND prediction_id=?',
            (user['id'], latest_pred['id'])
        ).fetchone()
        if existing:
            conn.close()
            return jsonify({'status':'error','message':'You already submitted a survey for your latest analysis.'}), 400

        conn.execute('''INSERT INTO surveys
            (user_id,prediction_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,avg_score,submitted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (user['id'], latest_pred['id'], *answers, avg, datetime.now().isoformat()))
        conn.execute('UPDATE users SET survey_done=1 WHERE id=?', (user['id'],))
        conn.commit()
        conn.close()
        return jsonify({'status':'success','message':'Thank you! Survey submitted.','redirect':'/dashboard'})
    except Exception as e:
        conn.close()
        return jsonify({'status':'error','message':str(e)}), 500

@app.route('/api/survey-stats')
def survey_stats_public():
    """Public endpoint — aggregated survey stats for the homepage charts."""
    conn = get_db()
    rows = conn.execute('SELECT q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,avg_score FROM surveys').fetchall()
    total = len(rows)
    conn.close()
    if total == 0:
        return jsonify({'total': 0, 'per_question': [], 'distribution': {}, 'overall_avg': 0})

    questions = [
        'Easy to navigate',
        'Clear AI results',
        'Useful diet & exercise plans',
        'Accurate medical info',
        'Simple data entry',
        'Good UI design',
        'Fast & responsive',
        'Reliable (no errors)',
        'Secure & private',
        'Overall satisfaction',
    ]

    per_question = []
    for i, label in enumerate(questions, 1):
        col = f'q{i}'
        avg = round(sum(r[col] for r in rows) / total, 2)
        per_question.append({'question': label, 'avg': avg, 'index': i})

    # Distribution of ratings 1-5 across ALL answers
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in rows:
        for i in range(1, 11):
            dist[r[f'q{i}']] += 1

    overall = round(sum(r['avg_score'] for r in rows) / total, 2)
    return jsonify({
        'total': total,
        'overall_avg': overall,
        'per_question': per_question,
        'distribution': dist,
    })


@app.route('/survey/results')
def survey_results():
    """Admin view — all survey results summary."""
    if 'user_id' not in session:
        return redirect('/login')
    conn = get_db()
    rows = conn.execute('''
        SELECT s.*, u.username, u.email
        FROM surveys s JOIN users u ON s.user_id=u.id
        ORDER BY s.submitted_at DESC
    ''').fetchall()
    total = len(rows)
    if total == 0:
        conn.close()
        return jsonify({'total':0,'message':'No surveys yet'})
    avgs = {f'q{i}': round(sum(r[f'q{i}'] for r in rows)/total,2) for i in range(1,11)}
    overall = round(sum(r['avg_score'] for r in rows)/total, 2)
    conn.close()
    return jsonify({
        'total_responses': total,
        'overall_avg': overall,
        'per_question_avg': avgs,
        'responses': [dict(r) for r in rows]
    })


# ─── Ollama Chatbot Route ─────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    """
    DiabetesAI chatbot endpoint using local Ollama.

    Requirements:
    - Ollama must be installed and running.
    - The model must be pulled once:
      ollama pull qwen2.5:3b

    Behavior:
    - Arabic question => Arabic answer.
    - English question => English answer.
    - Tolerates small spelling mistakes.
    - If the question is vague inside DiabetesAI, assume it refers to diabetes/platform.
    - Rejects only clearly unrelated or mostly unreadable questions.
    """
    try:
        print("\n========== OLLAMA CHAT ROUTE CALLED ==========")

        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        print("USER MESSAGE:", message)

        if not message:
            return jsonify({'status': 'error', 'message': 'Empty message'}), 400

        import urllib.request
        import urllib.error
        import json as _json
        import re
        import difflib

        ollama_model = "qwen2.5:3b"
        ollama_url = "http://127.0.0.1:11434/api/generate"

        # ─── Language & text helpers ───────────────────────────────────────────
        def has_arabic(text):
            return bool(re.search(r'[\u0600-\u06FF]', text))

        def normalize_text(text):
            text = (text or "").lower().strip()
            text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            text = text.replace("ى", "ي").replace("ة", "ه")
            text = text.replace("ؤ", "و").replace("ئ", "ي")
            text = text.replace("ـ", "")
            text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        def typo_match(word, keywords, cutoff=0.72):
            if not word or len(word) < 4:
                return False
            for k in keywords:
                nk = normalize_text(k)
                if len(nk) < 4:
                    continue
                ratio = difflib.SequenceMatcher(None, word, nk).ratio()
                if ratio >= cutoff:
                    return True
            return False

        def looks_mostly_unreadable(text):
            """
            Reject only when the text is mostly random/gibberish.
            We don't reject normal slang or spelling mistakes.
            """
            t = normalize_text(text)
            if not t:
                return True

            letters = re.findall(r'[a-zA-Z\u0600-\u06FF]', t)
            if len(letters) < 2:
                return True

            words = t.split()
            if len(words) == 1 and len(words[0]) <= 2:
                return True

            # Repeated random characters like "سسسسسسسس" or "xxxxxxx"
            for w in words:
                if len(w) >= 7 and len(set(w)) <= 2:
                    return True

            return False

        is_arabic = has_arabic(message)
        normalized_message = normalize_text(message)

        # ─── Wide project scope keywords ───────────────────────────────────────
        arabic_keywords = [
            # Diabetes names and common forms
            "سكري", "السكري", "سكر", "السكر", "مرض السكر", "مرض السكري", "داء السكري",
            "ديابيتس", "دايبتس", "دايبتيز", "دايبتيز", "مريض سكر", "مريض", "المرض",
            "مرض", "مزمن", "مزمنه",

            # Symptoms / diagnosis / tests
            "اعراض", "اعراضه", "علامات", "اشارات", "تشخيص", "فحص", "تحليل", "تحاليل",
            "نتيجه", "نتائج", "قياس", "سكر صايم", "سكر فاطر", "تراكمي", "هيموجلوبين",
            "جلوكوز", "غلوكوز", "انسولين", "الانسولين", "ضغط", "ضغط الدم", "دم", "الدم",
            "كوليسترول", "دهون", "مقاومه", "مقاومه الانسولين", "جرعه", "دواء", "ادويه",
            "علاج", "اسباب", "سبب", "مضاعفات", "وقايه", "طبيب", "دكتور",

            # Nutrition / lifestyle
            "اكل", "اكلات", "طعام", "غذاء", "تغذيه", "نظام", "نظام غذائي", "دايت",
            "حميه", "وجبه", "فطار", "غدا", "عشا", "سعرات", "كالوري", "كارب",
            "كربوهيدرات", "نشويات", "بروتين", "دهون", "مسموح", "ممنوع", "ينفع اكل",
            "اشرب", "مياه", "رياضه", "تمرين", "تمارين", "مشي", "جري", "جيم",
            "نوم", "توتر", "قلق", "ضغط نفسي",

            # Body metrics
            "وزن", "الوزن", "تخسيس", "تخس", "سمنه", "نحافه", "طول", "كتله", "مؤشر",
            "عمر", "سن",

            # Platform/project
            "المشروع", "مشروع", "المنصه", "المنصة", "الموقع", "السيستم", "النظام",
            "التطبيق", "ابلكيشن", "ويب", "ديابيتس", "داشبورد", "الداشبورد",
            "تسجيل", "دخول", "لوجن", "حساب", "اكونت", "استبيان", "اسئله", "تقييم",
            "تحليل", "اناليسيس", "اقتراح", "اقتراحات", "خطة", "خطه", "نتيجتي",
            "تحليلي", "حساباتي",

            # AI / ML / dataset / backend
            "ذكاء", "ذكاء اصطناعي", "موديل", "الموديل", "نموذج", "النموذج",
            "توقع", "تنبؤ", "خطر", "نسبه", "احتمال", "دقه", "الدقه", "داتا",
            "الداتا", "بيانات", "قاعده بيانات", "جدول", "باك", "باك اند", "فرونت",
            "فرونت اند", "فلاسك", "بايثون", "اولاما", "كوين", "جيميناي", "استضافه",
            "هوست", "رفع", "جيت هب"
        ]

        english_keywords = [
            # Diabetes/medical
            "diabetes", "diabetic", "diabete", "diabtes", "diabetees", "sugar",
            "blood sugar", "glucose", "insulin", "hba1c", "bmi", "type 1", "type 2",
            "gestational", "prediabetes", "symptom", "symptoms", "sign", "signs",
            "cause", "causes", "diagnosis", "diagnose", "test", "tests", "treatment",
            "medicine", "medication", "doctor", "complication", "complications",
            "prevention", "prevent", "risk", "blood", "pressure", "obesity", "weight",
            "age", "cholesterol", "insulin resistance",

            # Nutrition/lifestyle
            "diet", "nutrition", "food", "meal", "meal plan", "calorie", "calories",
            "carb", "carbs", "protein", "fat", "fats", "exercise", "workout", "walking",
            "running", "sleep", "stress", "monitoring", "healthy", "unhealthy",

            # Project/platform/tools
            "project", "platform", "website", "web app", "application", "app", "system",
            "diabetesai", "diabetes ai", "dashboard", "login", "register", "account",
            "survey", "analysis", "result", "results", "recommendation", "recommendations",
            "prediction", "predict", "model", "ai", "accuracy", "dataset", "data",
            "database", "sqlite", "csv", "pkl", "flask", "python", "html", "css",
            "javascript", "js", "bootstrap", "api", "route", "post", "get", "backend",
            "frontend", "ollama", "qwen", "gemini", "hosting", "deployment", "deploy",
            "pythonanywhere", "render", "github", "chatbot"
        ]

        # Very broad vague phrases that are allowed because user is inside DiabetesAI
        vague_arabic_phrases = [
            "ما هو", "ما هي", "ايه هو", "اي هو", "يعني ايه", "يعني اي", "اشرح",
            "فهمني", "وضح", "قولي", "ازاي", "ليه", "ينفع", "ممكن", "اعمل اي",
            "اعمل ايه", "اتعامل ازاي", "استخدمه ازاي", "ده ايه", "دي ايه", "دا ايه",
            "ما هو المرض", "ايه المرض", "ما العلاج", "ايه العلاج", "ايه الاعراض",
            "ما الاعراض", "اكل اي", "الخطة", "خطه", "النظام المناسب", "هل انا",
            "انا عندي", "عندي", "حاسس", "حاسة", "ايه النتيجه", "نتيجتي معناها ايه"
        ]

        vague_english_phrases = [
            "what is", "what are", "explain", "tell me", "how to", "how can",
            "can i", "should i", "is it", "why", "what should", "what does",
            "meaning of", "help me", "my result", "my analysis", "my plan",
            "my diet", "what is the disease", "what are symptoms", "what is treatment"
        ]

        def is_allowed_question(text):
            """
            Very tolerant local filter:
            - Allows any clear or vague hint about diabetes/project.
            - Allows spelling mistakes if most of the message is understandable.
            - Rejects only clearly unrelated topics or mostly unreadable text.
            """
            t = normalize_text(text)
            words = t.split()

            if looks_mostly_unreadable(text):
                return False

            all_keywords = arabic_keywords + english_keywords

            # Direct keyword/phrase match
            for k in all_keywords:
                nk = normalize_text(k)
                if nk and nk in t:
                    return True

            # Typo tolerance for both Arabic and English keywords
            for w in words:
                if len(w) >= 4:
                    if typo_match(w, english_keywords, cutoff=0.70):
                        return True
                    if typo_match(w, arabic_keywords, cutoff=0.74):
                        return True

            # Vague questions are allowed inside this medical/project assistant
            for phrase in vague_arabic_phrases + vague_english_phrases:
                if normalize_text(phrase) in t:
                    return True

            # Mixed short words like "sugar high", "model?", "survey?"
            if len(words) <= 4:
                short_allowed = ["sugar", "model", "survey", "diet", "insulin", "glucose", "bmi",
                                 "سكر", "نظام", "اكل", "موديل", "استبيان", "تحليل", "مرض"]
                for w in words:
                    if w in short_allowed:
                        return True

            return False

        def clearly_unrelated(text):
            """
            Reject only if the message clearly talks about something outside the project.
            If it is vague, we prefer answering in DiabetesAI context.
            """
            t = normalize_text(text)
            unrelated_keywords = [
                # Arabic unrelated
                "دولار", "سعر الدولار", "اليورو", "الذهب", "الدهب", "ماتش", "مباراه",
                "كوره", "كرة", "اغنيه", "اغاني", "فيلم", "مسلسل", "سياسه", "انتخابات",
                "طقس", "جو النهارده", "اخبار", "خبر", "بورصه", "اسهم", "بيتكوين",
                "كريبتو", "نكتة", "نكت", "قصة حب",
                # English unrelated
                "dollar", "currency", "euro", "gold price", "football", "match", "movie",
                "song", "weather", "news", "politics", "stock", "bitcoin", "crypto"
            ]
            return any(normalize_text(k) in t for k in unrelated_keywords)

        if clearly_unrelated(message) and not is_allowed_question(message):
            if is_arabic:
                return jsonify({
                    'status': 'success',
                    'reply': 'السؤال ده خارج نطاق مشروع DiabetesAI. أقدر أساعدك في السكري أو المنصة أو أدوات المشروع فقط.'
                }), 200
            return jsonify({
                'status': 'success',
                'reply': 'This question is outside the scope of DiabetesAI. I can help only with diabetes, the platform, or the project tools.'
            }), 200

        if not is_allowed_question(message):
            # Mostly unreadable / no meaningful project hint
            if is_arabic:
                return jsonify({
                    'status': 'success',
                    'reply': 'مش قادر أفهم السؤال بوضوح. اكتب سؤالك عن السكري أو تحليل الموقع أو المنصة بكلمات أوضح شوية.'
                }), 200
            return jsonify({
                'status': 'success',
                'reply': 'I could not understand the question clearly. Please ask about diabetes, the analysis, or the DiabetesAI platform more clearly.'
            }), 200

        language_instruction = (
            "The user wrote in Arabic. Reply in Arabic only. Use clear simple Arabic or Egyptian Arabic."
            if is_arabic else
            "The user wrote in English. Reply in English only."
        )

        # Strong prompt for accuracy
        prompt = f"""
You are DiabetesAI Assistant inside a diabetes prediction web project.

Context:
The user is using a platform called DiabetesAI. The platform predicts diabetes risk, shows health analysis,
suggests nutrition/exercise/monitoring advice, has login/register/dashboard/survey, and uses a machine learning model.

Accuracy rules:
- If the user asks a vague question like "ما هو المرض؟", "ما هو؟", "ايه الاعراض؟", "what is it?",
  assume they mean diabetes mellitus because they are inside DiabetesAI.
- Do NOT answer generally about all diseases unless the user explicitly asks generally.
- Focus on diabetes or the DiabetesAI platform.
- Be tolerant of spelling mistakes, slang, informal wording, mixed Arabic/English, and small typos.
- If a word is misspelled but the intent is likely diabetes/project, answer normally.
- Only refuse if the question is clearly unrelated to diabetes, health analysis, nutrition, exercise, the AI model, the survey,
  the dashboard, the database, or project tools.
- If the question asks for medical advice, give safe general information and remind the user to consult a doctor.

Allowed topics:
1) Diabetes: definition, type 1, type 2, gestational diabetes, symptoms, causes, diagnosis, treatment,
   complications, blood glucose, HbA1c, insulin, BMI, blood pressure.
2) Lifestyle: diet, nutrition, calories, carbs, protein, exercise, sleep, stress, monitoring.
3) DiabetesAI platform: prediction result, risk level, dashboard, login/register, survey, analysis form,
   dataset, database, model accuracy, diet recommendations, chatbot.
4) Project tools: Flask, Python, HTML, CSS, JavaScript, SQLite, ML model, Ollama, Qwen, Gemini,
   deployment, hosting, GitHub, PythonAnywhere.

Language:
{language_instruction}
Do not mix languages unless the user mixes them.

Answer style:
- Short, clear, and practical.
- For Arabic answers, avoid broken Arabic and avoid adding English terms unless necessary.
- For English answers, use simple English.
- If the question is vague, start by clarifying the assumed meaning briefly.
- Avoid hallucinating live/current facts.

User question:
{message}
"""

        body = _json.dumps({
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "num_predict": 280,
                "top_p": 0.85
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            ollama_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            print("OLLAMA RAW RESPONSE START:", raw[:500])
            result = _json.loads(raw)

        reply = result.get("response", "").strip()

        if not reply:
            print("OLLAMA EMPTY RESPONSE:", result)
            if is_arabic:
                return jsonify({'status': 'error', 'message': 'Ollama رد فاضي. جرّب السؤال مرة أخرى.'}), 200
            return jsonify({'status': 'error', 'message': 'Ollama returned an empty response. Please try again.'}), 200

        print("OLLAMA CHAT SUCCESS")
        return jsonify({'status': 'success', 'reply': reply})

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        print("OLLAMA HTTP ERROR CODE:", e.code)
        print("OLLAMA HTTP ERROR BODY:", err_body)
        return jsonify({
            'status': 'error',
            'message': 'Ollama رجّع خطأ. راجع التيرمنال عند OLLAMA HTTP ERROR BODY.'
        }), 200

    except urllib.error.URLError as e:
        print("OLLAMA CONNECTION ERROR:", str(e))
        return jsonify({
            'status': 'error',
            'message': 'Ollama غير شغال. افتح Ollama أو اكتب ollama serve، وتأكد إن موديل qwen2.5:3b متسطب.'
        }), 200

    except Exception as e:
        print('OLLAMA CHAT ERROR:', str(e))
        return jsonify({
            'status': 'error',
            'message': 'حدث خطأ في المساعد الذكي. راجع التيرمنال عند OLLAMA CHAT ERROR.'
        }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




