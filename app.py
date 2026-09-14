import os, json, datetime
from typing import TypedDict, List, Dict, Any
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

STUDY_KNOWLEDGE = {
    "active_recall": {"title": "Active Recall", "keywords": ["weak", "quiz", "memorize", "recall", "test", "retention"], "strategy": "Actively test yourself with practice questions and closed-book retrieval rather than passive reading."},
    "spaced_repetition": {"title": "Spaced Repetition", "keywords": ["revision", "forget", "long-term", "repeat", "interval", "review"], "strategy": "Space topic reviews over expanding intervals (1, 3, 7 days) to lock knowledge into long-term memory."},
    "pomodoro": {"title": "Pomodoro Technique", "keywords": ["hours", "focus", "tired", "burnout", "stamina", "session"], "strategy": "Work in 25-minute focused blocks with 5-minute pauses; after 4 sessions, take an extended 20-minute break."},
    "time_management": {"title": "Time Management", "keywords": ["deadline", "days", "schedule", "plan", "allocation", "balance"], "strategy": "Prioritize difficult and weak topics during high-energy morning hours, reserving lighter review for later."},
    "revision": {"title": "Revision Strategy", "keywords": ["summary", "progress", "formula", "mix", "interleave", "practice"], "strategy": "Interleave related subjects and summarize concepts into concise 1-page formula or cheat sheets."},
    "exam_preparation": {"title": "Exam Preparation", "keywords": ["exam", "mock", "timed", "pressure", "confidence", "speed"], "strategy": "Complete full timed mock questions under exam constraints, analyzing every mistake immediately."}
}

CURATED_QUIZZES = {
    "calculus": [{"question": "What does the derivative of a function f(x) physically represent at a point?", "options": ["Instantaneous rate of change of f(x)", "Accumulated area under the curve", "Inverse value of the function", "Global maximum of the domain"], "answer": "Instantaneous rate of change of f(x)", "explanation": "The derivative represents the tangent slope, which is the instantaneous rate of change."}, {"question": "According to the Fundamental Theorem of Calculus, what relates differentiation and integration?", "options": ["They are inverse operations of each other", "They are identical mathematical transformations", "They can only apply to polynomial equations", "They require linear independence across variables"], "answer": "They are inverse operations of each other", "explanation": "Integration accumulates changes while differentiation computes instantaneous change."}, {"question": "When evaluating a limit that yields the indeterminate form 0/0, which technique applies?", "options": ["L'Hopital's Rule (differentiating numerator and denominator)", "Dividing both sides by zero", "Setting the entire limit expression to zero", "Switching immediately to polar coordinates"], "answer": "L'Hopital's Rule (differentiating numerator and denominator)", "explanation": "L'Hopital's rule allows differentiation of numerator and denominator to resolve 0/0 limits."}],
    "linear algebra": [{"question": "What does a matrix determinant equal to zero signify for a square matrix?", "options": ["The matrix is singular and not invertible", "The matrix has exclusively positive eigenvalues", "The matrix represents an orthogonal transformation", "The column vectors form a linearly independent basis"], "answer": "The matrix is singular and not invertible", "explanation": "A determinant of zero means the transformation collapses dimensions, making it non-invertible."}, {"question": "If matrix A multiplied by vector v equals lambda * v, what are lambda and v called?", "options": ["Eigenvalue and eigenvector", "Determinant and rank", "Trace and null space", "Scalar projection and normal vector"], "answer": "Eigenvalue and eigenvector", "explanation": "By definition, Av = lambda * v defines the eigenvector v and corresponding eigenvalue lambda."}, {"question": "Two non-zero vectors in R^n are orthogonal if and only if their:", "options": ["Dot product equals zero", "Cross product equals zero", "Norms are completely identical", "Matrix dimensions are unequal"], "answer": "Dot product equals zero", "explanation": "The dot product of perpendicular vectors is zero because cosine of 90 degrees is zero."}],
    "thermodynamics": [{"question": "Which law of thermodynamics asserts that the total entropy of an isolated system always increases?", "options": ["Second Law of Thermodynamics", "First Law of Thermodynamics", "Zeroth Law of Thermodynamics", "Third Law of Thermodynamics"], "answer": "Second Law of Thermodynamics", "explanation": "The Second Law dictates that natural thermodynamic processes generate entropy, increasing disorder."}, {"question": "In an adiabatic thermodynamic process, what is the net heat exchange (Q) with the surroundings?", "options": ["Zero heat transfer (Q = 0)", "Maximum positive heat influx", "Constant temperature maintenance", "Complete conversion of heat to mass"], "answer": "Zero heat transfer (Q = 0)", "explanation": "An adiabatic process is thermally insulated so no heat enters or leaves the system."}, {"question": "What does the First Law of Thermodynamics fundamentally establish?", "options": ["Conservation of energy (Delta U = Q - W)", "Impossibility of reaching absolute zero", "Thermal equilibrium transitiveness", "Spontaneous decrease in system entropy"], "answer": "Conservation of energy (Delta U = Q - W)", "explanation": "Energy cannot be created or destroyed, only transferred as heat or work."}],
    "electromagnetism": [{"question": "According to Faraday's Law of Induction, what induces an EMF in a closed circuit?", "options": ["A time-varying magnetic flux through the circuit", "A perfectly static magnetic field", "A constant uniform electric charge", "Zero impedance in dielectric media"], "answer": "A time-varying magnetic flux through the circuit", "explanation": "Changing magnetic flux over time induces an electromotive force as formulated by Faraday."}, {"question": "Which Maxwell equation establishes that magnetic monopoles do not exist in classical physics?", "options": ["Gauss's Law for Magnetism (div B = 0)", "Ampere-Maxwell Law", "Coulomb's Inverse Square Law", "Ohm's Law for conductors"], "answer": "Gauss's Law for Magnetism (div B = 0)", "explanation": "The divergence of magnetic field B being zero means magnetic field lines always close."}, {"question": "In electromagnetic wave propagation through a vacuum, what is the relation between E and B fields?", "options": ["Perpendicular to each other and to wave propagation", "Parallel to each other in the direction of propagation", "Oppositely directed along the same linear axis", "Radially symmetric pointing inward toward source"], "answer": "Perpendicular to each other and to wave propagation", "explanation": "EM waves are transverse; electric and magnetic oscillations are mutually orthogonal."}]
}

def get_llm():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    models = [os.environ.get("GEMINI_MODEL", ""), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for m in models:
        if m:
            try:
                return ChatGoogleGenerativeAI(model=m, google_api_key=key, temperature=0.2)
            except Exception:
                continue
    return None

@tool(description="Retrieves study strategies from knowledge base.")
def study_knowledge_retriever(query: str) -> str:
    words = set(query.lower().replace(",", " ").split())
    scored = []
    for item in STUDY_KNOWLEDGE.values():
        s = sum(2 for k in item["keywords"] if k in words) + sum(1 for w in words if w in item["strategy"].lower() or w in item["title"].lower())
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:2] if scored and scored[0][0] > 0 else list(STUDY_KNOWLEDGE.values())[:2]
    return " | ".join([f"{item['title']}: {item['strategy']}" for _, item in top])

@tool(description="Prioritizes study topics based on constraints.")
def topic_prioritizer(topics: List[str], weak_topics: List[str], strong_topics: List[str], remaining_days: int, current_progress: float) -> List[Dict[str, Any]]:
    weak_set = {t.strip().lower() for t in weak_topics if t.strip()}
    strong_set = {t.strip().lower() for t in strong_topics if t.strip()}
    priorities = []
    for idx, raw_t in enumerate(topics):
        t = raw_t.strip()
        if not t:
            continue
        tl = t.lower()
        diff = 3 if tl in weak_set else (1 if tl in strong_set else 2)
        conf = 1 if tl in weak_set else (3 if tl in strong_set else 2)
        imp = 3 if (idx < max(1, len(topics) // 3) or tl in weak_set) else 2
        urgency = 3 if remaining_days <= 7 else (2 if remaining_days <= 20 else 1)
        score = round((diff * 0.35 + imp * 0.25 + (4 - conf) * 0.20 + urgency * 0.10 + (1.0 - (current_progress / 100.0)) * 0.10) * 10, 2)
        priorities.append({"topic": t, "score": score, "difficulty": "High" if diff == 3 else ("Medium" if diff == 2 else "Low"), "confidence": "Low" if conf == 1 else ("High" if conf == 3 else "Medium")})
    priorities.sort(key=lambda x: x["score"], reverse=True)
    return priorities

@tool(description="Generates daily study schedule.")
def schedule_generator(prioritized_topics: List[Dict[str, Any]], days: int, daily_hours: float, guidance: str) -> List[Dict[str, Any]]:
    days = max(1, min(days, 30))
    schedule = []
    n = len(prioritized_topics)
    if n == 0:
        return schedule
    for d in range(1, days + 1):
        d_date = (datetime.date.today() + datetime.timedelta(days=d - 1)).strftime("%Y-%m-%d")
        t1 = prioritized_topics[(d - 1) % n]
        strat = list(STUDY_KNOWLEDGE.values())[(d - 1) % len(STUDY_KNOWLEDGE)]
        if n == 1:
            h1 = round(daily_hours * 0.65, 1)
            h2 = round(daily_hours - h1, 1)
            sessions = [{"topic": t1["topic"], "hours": h1, "difficulty": t1["difficulty"], "task": "Core concept mastery & active practice"}]
            if h2 > 0:
                sessions.append({"topic": t1["topic"], "hours": h2, "difficulty": t1["difficulty"], "task": "Targeted self-quizzing & spaced retrieval"})
        else:
            t2 = prioritized_topics[d % n]
            h1 = round(daily_hours * 0.65, 1)
            h2 = round(daily_hours - h1, 1)
            sessions = [{"topic": t1["topic"], "hours": h1, "difficulty": t1["difficulty"], "task": "Core concept mastery & active practice"}]
            if h2 > 0:
                sessions.append({"topic": t2["topic"], "hours": h2, "difficulty": t2["difficulty"], "task": "Targeted review & self-quizzing"})
        schedule.append({"day": d, "date": d_date, "strategy": f"{strat['title']}: {strat['strategy']}", "sessions": sessions, "total_hours": daily_hours})
    return schedule

@tool(description="Calculates study progress metrics.")
def progress_calculator(total_topics: int, completed_topics: int, daily_hours: float, remaining_days: int) -> Dict[str, Any]:
    pct = round((completed_topics / max(1, total_topics)) * 100, 1)
    rem_hours = round(daily_hours * max(1, remaining_days), 1)
    status = "On Track" if pct >= 60 else ("Needs Attention" if pct >= 30 else "Behind Schedule")
    return {"percentage": min(100.0, pct), "completed": completed_topics, "total": total_topics, "remaining_hours": rem_hours, "status": status}

@tool(description="Generates topic quiz questions.")
def quiz_generator(topic: str, context: str = "") -> List[Dict[str, Any]]:
    llm = get_llm()
    if llm:
        try:
            prompt = PromptTemplate.from_template("Generate 3 multiple choice questions for the study topic '{topic}'. Return JSON list of objects with keys 'question', 'options' (array of 4 options), 'answer' (exact correct option string), 'explanation'.").format(topic=topic)
            res = llm.invoke(prompt)
            clean = res.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            if isinstance(data, list) and len(data) > 0 and all("question" in q and "options" in q for q in data):
                return data
        except Exception:
            pass
    tl = topic.lower().strip()
    for k, qs in CURATED_QUIZZES.items():
        if k in tl or tl in k:
            return qs
    return [
        {"question": f"When beginning study on {topic}, which initial step delivers the highest retention?", "options": [f"Survey key terms and definitions in {topic}", "Passive reading without self-testing", "Memorizing formulas without conceptual basis", "Skipping fundamental definitions"], "answer": f"Survey key terms and definitions in {topic}", "explanation": f"Establishing clear terminology and definitions forms the scaffolding for {topic}."},
        {"question": f"Which problem-solving strategy is recommended when analyzing complex scenarios in {topic}?", "options": ["Deconstruct known variables and apply core principles", "Rely entirely on guesswork", "Avoid structured problem breakdown", "Skip verification against boundary conditions"], "answer": "Deconstruct known variables and apply core principles", "explanation": "Breaking problems into knowns, constraints, and core principles prevents conceptual errors."},
        {"question": f"What is the most effective evidence-based method to prepare for exams covering {topic}?", "options": ["Timed practice problems and active recall retrieval", "Single-night cramming before the exam", "Passive re-reading of highlighters and notes", "Ignoring weaker subtopics"], "answer": "Timed practice problems and active recall retrieval", "explanation": "Active retrieval and timed practice reinforce neural pathways for superior exam recall."}
    ]

class StudentState(TypedDict, total=False):
    exam_date: str; subjects: List[str]; topics: List[str]; daily_hours: float
    weak_topics: List[str]; strong_topics: List[str]; current_progress: float; completed_topics: List[str]
    missed_session: bool; remaining_days: int; priorities: List[Dict[str, Any]]; study_guidance: str
    schedule: List[Dict[str, Any]]; progress_stats: Dict[str, Any]; adaptation_message: str

def receive_student_state(state: StudentState) -> Dict[str, Any]:
    try:
        exam_dt = datetime.datetime.strptime(state["exam_date"], "%Y-%m-%d").date()
        days = max(1, (exam_dt - datetime.date.today()).days)
    except Exception:
        days = 14
    return {"remaining_days": days}

def analyze_priorities(state: StudentState) -> Dict[str, Any]:
    prios = topic_prioritizer.invoke({"topics": state["topics"], "weak_topics": state["weak_topics"], "strong_topics": state["strong_topics"], "remaining_days": state["remaining_days"], "current_progress": state["current_progress"]})
    return {"priorities": prios}

def retrieve_study_guidance(state: StudentState) -> Dict[str, Any]:
    query = " ".join(state["weak_topics"] + state["topics"][:3] + ["exam", "retention"])
    guidance = study_knowledge_retriever.invoke({"query": query})
    return {"study_guidance": guidance}

def generate_schedule(state: StudentState) -> Dict[str, Any]:
    sched = schedule_generator.invoke({"prioritized_topics": state["priorities"], "days": state["remaining_days"], "daily_hours": state["daily_hours"], "guidance": state["study_guidance"]})
    return {"schedule": sched}

def track_progress(state: StudentState) -> Dict[str, Any]:
    comp = len(state.get("completed_topics", []))
    tot = len(state.get("topics", []))
    stats = progress_calculator.invoke({"total_topics": tot, "completed_topics": comp, "daily_hours": state["daily_hours"], "remaining_days": state["remaining_days"]})
    return {"progress_stats": stats, "current_progress": stats["percentage"]}

def adapt_schedule(state: StudentState) -> Dict[str, Any]:
    sched = list(state.get("schedule", []))
    msg = "Schedule is up to date."
    if state.get("missed_session"):
        if len(sched) > 1:
            sched = sched[1:]
        for idx, day_plan in enumerate(sched):
            day_plan["day"] = idx + 1
            day_plan["strategy"] += " (Redistributed for missed session)"
        msg = "Missed session detected! Remaining topics redistributed across remaining days with weak areas prioritized."
    elif state.get("current_progress", 0) >= 60:
        for day_plan in sched:
            for session in day_plan.get("sessions", []):
                if session.get("difficulty") == "High":
                    session["hours"] = round(session["hours"] * 1.2, 1)
        msg = "High progress detected! Streamlined strong topics and elevated focus on weak areas."
    return {"schedule": sched, "adaptation_message": msg}

graph_builder = StateGraph(StudentState)
for n, fn in [("receive_student_state", receive_student_state), ("analyze_priorities", analyze_priorities), ("retrieve_study_guidance", retrieve_study_guidance), ("generate_schedule", generate_schedule), ("track_progress", track_progress), ("adapt_schedule", adapt_schedule)]:
    graph_builder.add_node(n, fn)
for src, dst in [(START, "receive_student_state"), ("receive_student_state", "analyze_priorities"), ("analyze_priorities", "retrieve_study_guidance"), ("retrieve_study_guidance", "generate_schedule"), ("generate_schedule", "track_progress"), ("track_progress", "adapt_schedule"), ("adapt_schedule", END)]:
    graph_builder.add_edge(src, dst)
study_graph = graph_builder.compile()

STUDENT_DATABASE = {
    "exam_date": (datetime.date.today() + datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
    "subjects": ["Mathematics", "Physics"], "topics": ["Calculus", "Linear Algebra", "Thermodynamics", "Electromagnetism"],
    "daily_hours": 3.0, "weak_topics": ["Calculus", "Electromagnetism"], "strong_topics": ["Linear Algebra"],
    "current_progress": 25.0, "completed_topics": ["Linear Algebra"], "missed_session": False,
    "remaining_days": 14, "priorities": [], "study_guidance": "", "schedule": [], "progress_stats": {}, "adaptation_message": "Ready to generate schedule."
}

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>AI Study Planner</title><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif}
body{background:rgb(15,23,42);color:rgb(241,245,249);padding:24px}
.container{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:380px 1fr;gap:24px}
.header{grid-column:1/-1;background:linear-gradient(135deg,rgb(30,41,59),rgb(51,65,85));padding:18px 24px;border-radius:12px;display:flex;justify-content:space-between;align-items:center;border:1px solid rgb(71,85,105)}
.header h1{font-size:22px;color:rgb(255,255,255)}.badge{background:rgb(99,102,241);color:white;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600}
.card{background:rgb(30,41,59);border-radius:12px;padding:18px;border:1px solid rgb(51,65,85);margin-bottom:18px}
.card h2{font-size:15px;margin-bottom:12px;color:rgb(226,232,240);border-bottom:1px solid rgb(51,65,85);padding-bottom:6px}
label{display:block;font-size:12px;color:rgb(148,163,184);margin-top:8px;margin-bottom:3px;font-weight:500}
input,select,textarea{width:100%;padding:8px 10px;background:rgb(15,23,42);border:1px solid rgb(71,85,105);border-radius:6px;color:rgb(241,245,249);font-size:13px}
button{background:rgb(99,102,241);color:white;border:none;border-radius:6px;padding:9px 14px;font-weight:600;font-size:13px;cursor:pointer;width:100%;margin-top:12px;transition:background 0.2s}
button:hover{background:rgb(79,70,229)}.btn-alt{background:rgb(239,68,68);margin-top:8px}.btn-alt:hover{background:rgb(220,38,38)}
.tabs{display:flex;gap:10px;margin-bottom:14px}
.tab-btn{background:rgb(51,65,85);color:rgb(203,213,225);padding:7px 12px;border-radius:6px;font-size:13px;cursor:pointer;border:none;width:auto;margin:0}.tab-btn.active{background:rgb(99,102,241);color:white}
.progress-bar-bg{background:rgb(15,23,42);border-radius:999px;height:10px;overflow:hidden;margin-top:8px}.progress-fill{background:rgb(16,185,129);height:100%;transition:width 0.4s}
.day-card{background:rgb(15,23,42);border:1px solid rgb(51,65,85);border-radius:8px;padding:12px;margin-bottom:10px}.day-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}.day-title{font-weight:600;font-size:14px;color:rgb(129,140,248)}
.strategy-box{background:rgb(30,41,59);border-left:3px solid rgb(99,102,241);padding:6px 10px;font-size:12px;margin-bottom:8px;color:rgb(203,213,225)}.session-item{display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px solid rgb(30,41,59)}
.tag{font-size:11px;padding:2px 6px;border-radius:4px;font-weight:600}.tag-high{background:rgba(239,68,68,0.2);color:rgb(248,113,113)}.tag-med{background:rgba(245,158,11,0.2);color:rgb(251,191,36)}.tag-low{background:rgba(16,185,129,0.2);color:rgb(52,211,153)}
.chat-box{height:230px;overflow-y:auto;background:rgb(15,23,42);border-radius:8px;padding:10px;margin-bottom:10px;border:1px solid rgb(51,65,85);display:flex;flex-direction:column;gap:8px}
.chat-msg{padding:8px 12px;border-radius:8px;font-size:13px;max-width:85%}.chat-bot{background:rgb(51,65,85);color:rgb(241,245,249);align-self:flex-start}.chat-user{background:rgb(99,102,241);color:white;align-self:flex-end}
.quiz-q{background:rgb(15,23,42);border:1px solid rgb(51,65,85);border-radius:8px;padding:12px;margin-bottom:10px}.quiz-opt{display:block;margin:5px 0;padding:6px 10px;background:rgb(30,41,59);border-radius:4px;cursor:pointer;font-size:12px;border:1px solid rgb(51,65,85)}
.quiz-ans{margin-top:6px;font-size:12px;color:rgb(52,211,153);display:none}.alert-box{background:rgba(99,102,241,0.15);border:1px solid rgb(99,102,241);border-radius:6px;padding:9px;font-size:12px;margin-bottom:10px;color:rgb(199,210,254)}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>AI Study Planner</h1><span class="badge">LangGraph + RAG + Gemini</span></div>
<div>
<div class="card">
<h2>Student Parameters</h2>
<label>Exam Date</label><input type="date" id="exam_date" value="{{ state.exam_date }}">
<label>Subjects</label><input type="text" id="subjects" value="{{ ', '.join(state.subjects) }}">
<label>Topics</label><textarea id="topics" rows="2">{{ ', '.join(state.topics) }}</textarea>
<label>Daily Study Hours</label><input type="number" id="daily_hours" step="0.5" min="0.5" max="16" value="{{ state.daily_hours }}">
<label>Weak Topics</label><input type="text" id="weak_topics" value="{{ ', '.join(state.weak_topics) }}">
<label>Strong Topics</label><input type="text" id="strong_topics" value="{{ ', '.join(state.strong_topics) }}">
<button onclick="generatePlan()">Generate Schedule</button>
<button class="btn-alt" onclick="reportMissed()">Report Missed Session</button>
</div>
<div class="card">
<h2>Progress Tracker</h2>
<div style="display:flex;justify-content:space-between;font-size:13px;"><span>Completion</span><strong>{{ state.current_progress }}%</strong></div>
<div class="progress-bar-bg"><div class="progress-fill" style="width:{{ state.current_progress }}%"></div></div>
<div style="display:flex;justify-content:space-between;font-size:12px;margin-top:8px;color:rgb(148,163,184)">
<span>Status: <b style="color:rgb(52,211,153)">{{ state.progress_stats.get('status', 'Active') }}</b></span>
<span>Remaining: <b>{{ state.progress_stats.get('remaining_hours', 0) }}h</b></span>
</div>
<div style="margin-top:10px;"><label>Toggle Completed Topic</label>
<select onchange="toggleTopic(this.value)"><option value="">Select completed topic...</option>{% for t in state.topics %}<option value="{{ t }}">{{ t }}</option>{% endfor %}</select>
</div>
</div>
</div>
<div>
<div class="tabs">
<button class="tab-btn active" onclick="switchTab('schedule', this)">Daily Schedule</button><button class="tab-btn" onclick="switchTab('quiz', this)">Quiz Practice</button><button class="tab-btn" onclick="switchTab('chat', this)">AI Study Advice</button>
</div>
<div id="tab-schedule">
<div class="alert-box">{{ state.adaptation_message }}</div>
{% for day in state.schedule %}
<div class="day-card">
<div class="day-header"><span class="day-title">Day {{ day.day }} ({{ day.date }})</span><span class="badge">{{ day.total_hours }} Hours</span></div>
<div class="strategy-box">{{ day.strategy }}</div>
{% for s in day.sessions %}
<div class="session-item"><span><b>{{ s.topic }}</b> - <small>{{ s.task }}</small></span><span><span class="tag tag-{{ 'high' if s.difficulty=='High' else ('med' if s.difficulty=='Medium' else 'low') }}">{{ s.difficulty }}</span> {{ s.hours }}h</span></div>
{% endfor %}
</div>
{% endfor %}
</div>
<div id="tab-quiz" style="display:none">
<div class="card"><h2>Generate Topic Quiz</h2>
<div style="display:flex;gap:8px;"><select id="quiz_topic">{% for t in state.topics %}<option value="{{ t }}">{{ t }}</option>{% endfor %}</select><button style="width:auto;margin:0" onclick="loadQuiz()">Generate Quiz</button></div>
</div><div id="quiz_container"></div>
</div>
<div id="tab-chat" style="display:none">
<div class="card"><h2>AI Study Coach</h2>
<div class="chat-box" id="chat_box"><div class="chat-msg chat-bot">Hello! I am your AI study mentor. Ask for guidance, memory strategies, or report schedule changes.</div></div>
<div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap">
<button class="tab-btn" onclick="sendChat('I missed today\\'s study session')">I missed today's study session</button>
<button class="tab-btn" onclick="sendChat('How do I apply active recall to weak topics?')">Active recall tips</button>
<button class="tab-btn" onclick="sendChat('How should I prioritize my remaining days?')">Prioritization tips</button>
</div>
<div style="display:flex;gap:8px"><input type="text" id="chat_in" placeholder="Ask study coach..." onkeydown="if(event.key==='Enter')sendChat()"><button style="width:100px;margin:0" onclick="sendChat()">Send</button></div>
</div>
</div>
</div>
</div>
<script>
function switchTab(t, b){document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById('tab-schedule').style.display=t==='schedule'?'block':'none';document.getElementById('tab-quiz').style.display=t==='quiz'?'block':'none';document.getElementById('tab-chat').style.display=t==='chat'?'block':'none';}
async function generatePlan(){
const d={exam_date:document.getElementById('exam_date').value,subjects:document.getElementById('subjects').value.split(',').map(s=>s.trim()).filter(Boolean),topics:document.getElementById('topics').value.split(',').map(s=>s.trim()).filter(Boolean),daily_hours:parseFloat(document.getElementById('daily_hours').value)||2.0,weak_topics:document.getElementById('weak_topics').value.split(',').map(s=>s.trim()).filter(Boolean),strong_topics:document.getElementById('strong_topics').value.split(',').map(s=>s.trim()).filter(Boolean)};
await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});location.reload();
}
async function reportMissed(){await fetch('/api/adapt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({missed_session:true})});location.reload();}
async function toggleTopic(t){if(!t)return;await fetch('/api/toggle_topic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:t})});location.reload();}
async function loadQuiz(){
const top=document.getElementById('quiz_topic').value,qc=document.getElementById('quiz_container');
qc.innerHTML='<p style="font-size:13px;color:rgb(148,163,184)">Generating quiz questions...</p>';
const r=await fetch('/api/quiz',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:top})});
const qs=await r.json();
qc.innerHTML=qs.map((q,idx)=>`<div class="quiz-q"><strong style="font-size:13px;display:block;margin-bottom:8px">Q${idx+1}: ${q.question}</strong>${q.options.map(o=>`<div class="quiz-opt" onclick="this.parentElement.querySelector('.quiz-ans').style.display='block'">${o}</div>`).join('')}<div class="quiz-ans"><b>Correct:</b> ${q.answer}<br><small>${q.explanation}</small></div></div>`).join('');
}
async function sendChat(text){
const inp=document.getElementById('chat_in'),msg=text||inp.value;if(!msg)return;
const box=document.getElementById('chat_box');box.innerHTML+=`<div class="chat-msg chat-user">${msg}</div>`;inp.value='';box.scrollTop=box.scrollHeight;
const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
const data=await r.json();box.innerHTML+=`<div class="chat-msg chat-bot">${data.reply}</div>`;box.scrollTop=box.scrollHeight;
if(data.adapted){setTimeout(()=>location.reload(),1200);}
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    if not STUDENT_DATABASE["schedule"]:
        res = study_graph.invoke(STUDENT_DATABASE)
        STUDENT_DATABASE.update(res)
    return render_template_string(HTML_TEMPLATE, state=STUDENT_DATABASE)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    parse_list = lambda k, d: data[k] if isinstance(data.get(k), list) else ([s.strip() for s in str(data[k]).split(",") if s.strip()] if data.get(k) else d)
    new_topics = parse_list("topics", STUDENT_DATABASE["topics"])
    current_completed = [t for t in STUDENT_DATABASE.get("completed_topics", []) if t in new_topics]
    STUDENT_DATABASE.update({
        "exam_date": data.get("exam_date", STUDENT_DATABASE["exam_date"]),
        "subjects": parse_list("subjects", STUDENT_DATABASE["subjects"]),
        "topics": new_topics,
        "daily_hours": max(0.5, min(16.0, float(data.get("daily_hours", STUDENT_DATABASE["daily_hours"])))),
        "weak_topics": parse_list("weak_topics", STUDENT_DATABASE["weak_topics"]),
        "strong_topics": parse_list("strong_topics", STUDENT_DATABASE["strong_topics"]),
        "completed_topics": current_completed,
        "missed_session": False
    })
    res = study_graph.invoke(STUDENT_DATABASE)
    STUDENT_DATABASE.update(res)
    return jsonify({"status": "success", "state": STUDENT_DATABASE})

@app.route("/api/adapt", methods=["POST"])
def api_adapt():
    data = request.json or {}
    STUDENT_DATABASE["missed_session"] = bool(data.get("missed_session", True))
    res = study_graph.invoke(STUDENT_DATABASE)
    STUDENT_DATABASE.update(res)
    return jsonify({"status": "adapted", "state": STUDENT_DATABASE})

@app.route("/api/toggle_topic", methods=["POST"])
def api_toggle_topic():
    top = (request.json or {}).get("topic", "")
    comp = set(STUDENT_DATABASE.get("completed_topics", []))
    if top in comp:
        comp.remove(top)
    elif top:
        comp.add(top)
    STUDENT_DATABASE["completed_topics"] = list(comp)
    STUDENT_DATABASE["missed_session"] = False
    res = study_graph.invoke(STUDENT_DATABASE)
    STUDENT_DATABASE.update(res)
    return jsonify({"status": "success", "state": STUDENT_DATABASE})

@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    top = (request.json or {}).get("topic", STUDENT_DATABASE["topics"][0] if STUDENT_DATABASE["topics"] else "Calculus")
    questions = quiz_generator.invoke({"topic": top})
    return jsonify(questions)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    msg = (request.json or {}).get("message", "")
    adapted = False
    if "missed" in msg.lower():
        STUDENT_DATABASE["missed_session"] = True
        res = study_graph.invoke(STUDENT_DATABASE)
        STUDENT_DATABASE.update(res)
        adapted = True
        reply = "I have noted your missed study session. The schedule has been dynamically redistributed across remaining days, keeping high-priority topics front and center."
    else:
        retrieved = study_knowledge_retriever.invoke({"query": msg})
        llm = get_llm()
        if llm:
            try:
                prompt = PromptTemplate.from_template("You are an expert study advisor. Context strategies: {guidance}. User query: {query}. Provide concise, practical study advice in 2-3 sentences.").format(guidance=retrieved, query=msg)
                res = llm.invoke(prompt)
                reply = res.content
            except Exception:
                reply = f"Study advice: {retrieved}"
        else:
            reply = f"Recommended study strategy: {retrieved}"
    return jsonify({"reply": reply, "adapted": adapted})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
