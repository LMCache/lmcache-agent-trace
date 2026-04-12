#!/usr/bin/env python3
"""Generate synthetic PyCodeAGI trace in JSONL format.

PyCodeAGI (https://github.com/chakkaradeep/pyCodeAGI) is a 5-step sequential
LLM pipeline (GPT-4 version: pycodeagi-gpt4.py) that generates Python/Streamlit
apps from a high-level objective.  Each step's *system message* accumulates all
prior outputs as a growing context block — a strong case for LMCache substring
cache reuse (CacheBlend).  Note: consecutive steps do NOT form a strict prefix
chain because each step changes its intro instruction line, breaking prefix
continuity at ~40 tokens.

Note: The original pycodeagi.py requires text-davinci-003 (deprecated Jan 2024).
This trace is based on the GPT-4 chat version (pycodeagi-gpt4.py), which cannot
be run in 2026 due to LangChain API changes (v0.0.139 → v0.3+).  This script
reproduces the prompt templates (adapted from pycodeagi-gpt4.py) so contributors
can generate real traces once the code is ported.

Usage:
    python generate_trace.py          # writes trace.jsonl in the same directory
    python generate_trace.py -o out.jsonl

Zero external dependencies — runs on any Python 3.7+.
"""

import hashlib
import json
import os
import argparse

# ---------------------------------------------------------------------------
# Prompt templates — adapted from pycodeagi-gpt4.py (ChatOpenAI / gpt-4)
# System message accumulates prior context; user message contains only the task.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_HEADER = (
    "You are code generation AI proficient in Python and Streamlit.\n"
    "Your goal is to build a Python app.\n"
    "You will use Streamlit for building the app user interface.\n"
    "Assume all required libraries are installed.\n"
    "{instructions}"
)

STEP_NAMES = ["description", "architecture", "ux_flow", "code_flow", "app_code"]

# sys_instructions → appended to system header (grows each step).
# user_task        → clean task instruction in [User] block.
STEPS_CONFIG = {
    "description": {
        "sys_instructions": (
            "Users will interact with the web app built using Streamlit and Python."
        ),
        "user_task": (
            "Create a concise description for the Python app: {objective}\n"
            "Use your expertise to envision the app's purpose and functionality."
        ),
    },
    "architecture": {
        "sys_instructions": (
            "You are given the app name and description.\n"
            "App Name:\n{objective}\n"
            "Description: \n{description}"
        ),
        "user_task": (
            "Create a concise app architecture you can use to build the UX flow.\n"
            "Outline the components and structure of the code.\n"
            "Present the app architecture in an ordered list."
        ),
    },
    "ux_flow": {
        "sys_instructions": (
            "You are given the app name, description and architecture.\n"
            "App Name:\n{objective}\n"
            "Description: \n{description}\n"
            "Architecture:\n{architecture}"
        ),
        "user_task": (
            "Create a concise UX flow that you can use to build code flow.\n"
            "Present the UX flow an ordered list."
        ),
    },
    "code_flow": {
        "sys_instructions": (
            "You are given the app name, description, architecture and UX flow.\n"
            "App Name:\n{objective}\n"
            "Description: \n{description}\n"
            "Architecture:\n{architecture}\n"
            "UX Flow:\n{ux_flow}"
        ),
        "user_task": (
            "Create a concise code flow you can use to write code.\n"
            "Outline the code components and structure.\n"
            "Present the code flow in an ordered list."
        ),
    },
    "app_code": {
        "sys_instructions": (
            "You are given the app name, description, architecture, UX flow and code flow.\n"
            "App Name:\n{objective}\n"
            "Description: \n{description}\n"
            "Architecture:\n{architecture}\n"
            "UX Flow:\n{ux_flow}\n"
            "Code Flow:\n{code_flow}"
        ),
        "user_task": (
            "Write the Python code for the app in a single python file.\n"
            "Use SQLite python module for data storage.\n"
            "Exclude environment setup, testing, debugging, and deployment tasks.\n"
            "Build sample datasets with at least five items.\n"
            "Follow these coding guidelines:\n"
            "- Check and create database tables first in the main function.\n"
            "- Use pd.loc to append new rows to the DataFrame.\n"
            "- When building date sliders: First Convert dates using to_pydatetime() "
            "then use their min and max values in st.slider.\n"
            "- Use pd.to_datetime() on selected date ranges when filtering calendar events.\n"
            "- Save all data in a SQLite database."
        ),
    },
}

# ---------------------------------------------------------------------------
# Synthetic outputs for each session
# ---------------------------------------------------------------------------

SYNTHETIC_SESSIONS = [
    {
        "objective": "calculator app",
        "outputs": {
            "description": (
                "A Streamlit-based Python calculator application that performs basic "
                "arithmetic operations (addition, subtraction, multiplication, division) "
                "with support for decimal numbers and operation history tracking. Users "
                "interact through a clean web interface with input fields and buttons."
            ),
            "architecture": (
                "Single-file Streamlit application using a modular functional design.\n"
                "Core components:\n"
                "1. Input Module — Two number inputs and an operator selector rendered "
                "via st.number_input and st.selectbox.\n"
                "2. Calculator Engine — Pure function that maps operator symbols to "
                "arithmetic operations, with explicit ZeroDivisionError handling.\n"
                "3. History Manager — Uses st.session_state with a collections.deque "
                "(maxlen=10) to persist calculation history across reruns.\n"
                "4. Display Module — Two-column layout: left column for input/result, "
                "right column for scrollable history."
            ),
            "ux_flow": (
                "1. App launches showing title 'Calculator App' and two-column layout\n"
                "2. Left column: user enters first number, selects operator (+, -, *, /), "
                "enters second number\n"
                "3. User clicks 'Calculate' button\n"
                "4. Result displayed as a success message below the button\n"
                "5. If division by zero, an error message is shown instead\n"
                "6. Each successful calculation is appended to the history deque\n"
                "7. Right column: history list updates in real time (most recent first)\n"
                "8. User can clear history with a 'Clear History' button"
            ),
            "code_flow": (
                "main() → st.title() → init_session_state() → render_input_column() → "
                "[st.number_input(num1) | st.selectbox(op) | st.number_input(num2)] → "
                "on_calculate_click() → calculate(num1, op, num2) → "
                "[display_result(result) | display_error()] → "
                "update_history(expr, result) → render_history_column() → "
                "[show_history() | clear_history()]"
            ),
            "app_code": (
                "import streamlit as st\n"
                "from collections import deque\n"
                "\n"
                "st.title('Calculator App')\n"
                "\n"
                "# Initialize history in session state\n"
                "if 'history' not in st.session_state:\n"
                "    st.session_state.history = deque(maxlen=10)\n"
                "\n"
                "col1, col2 = st.columns(2)\n"
                "\n"
                "with col1:\n"
                "    num1 = st.number_input('First number', value=0.0)\n"
                "    operator = st.selectbox('Operator', ['+', '-', '*', '/'])\n"
                "    num2 = st.number_input('Second number', value=0.0)\n"
                "\n"
                "    if st.button('Calculate'):\n"
                "        try:\n"
                "            ops = {\n"
                "                '+': num1 + num2,\n"
                "                '-': num1 - num2,\n"
                "                '*': num1 * num2,\n"
                "                '/': num1 / num2 if num2 != 0 else None,\n"
                "            }\n"
                "            result = ops[operator]\n"
                "            if result is None:\n"
                "                st.error('Cannot divide by zero!')\n"
                "            else:\n"
                "                st.success(f'Result: {result}')\n"
                "                st.session_state.history.append(\n"
                "                    f'{num1} {operator} {num2} = {result}'\n"
                "                )\n"
                "        except Exception as e:\n"
                "            st.error(f'Error: {e}')\n"
                "\n"
                "with col2:\n"
                "    st.subheader('History')\n"
                "    if st.button('Clear History'):\n"
                "        st.session_state.history.clear()\n"
                "    for entry in reversed(st.session_state.history):\n"
                "        st.text(entry)"
            ),
        },
    },
    {
        "objective": "todo list app",
        "outputs": {
            "description": (
                "A Streamlit-based todo list application that lets users create, read, "
                "update, and delete tasks. Each task has a title, optional description, "
                "priority level (High/Medium/Low), and completion status. Tasks are stored "
                "in session state and can be sorted by priority or creation order."
            ),
            "architecture": (
                "Single-file Streamlit application with dataclass-based task modeling.\n"
                "Core components:\n"
                "1. Task Model — Python dataclass with fields: id (uuid4), title, "
                "description, priority (enum: High/Medium/Low), done (bool), created_at.\n"
                "2. Task Store — List[Task] in st.session_state, providing add/remove/"
                "update/toggle operations.\n"
                "3. Filter & Sort Engine — Sidebar controls for filtering by status and "
                "sorting by priority or date.\n"
                "4. UI Layer — Streamlit form for adding tasks, expandable cards for "
                "existing tasks with inline edit/delete buttons."
            ),
            "ux_flow": (
                "1. App launches with title 'Todo List' and a sidebar for filters\n"
                "2. Sidebar: dropdown to filter (All / Active / Completed), sort selector\n"
                "3. Main area top: form with title input, description textarea, priority "
                "selector, and 'Add Task' button\n"
                "4. Below the form: list of task cards, each showing checkbox, title, "
                "priority badge, and expand button\n"
                "5. Expanding a card reveals description, created date, 'Edit' and "
                "'Delete' buttons\n"
                "6. Checking the checkbox toggles done status immediately\n"
                "7. 'Delete' removes the task with a confirmation toast\n"
                "8. Footer shows task count summary (e.g., '3 of 5 tasks completed')"
            ),
            "code_flow": (
                "main() → st.title() → init_tasks() → render_sidebar_filters() → "
                "render_add_task_form() → on_add_click() → Task(...) → "
                "append_to_store() → filter_tasks(status_filter) → "
                "sort_tasks(sort_key) → for task in filtered: render_task_card(task) → "
                "[toggle_done(task_id) | delete_task(task_id) | edit_task(task_id)] → "
                "render_footer_summary(total, completed)"
            ),
            "app_code": (
                "import streamlit as st\n"
                "import uuid\n"
                "from dataclasses import dataclass, field\n"
                "from datetime import datetime\n"
                "\n"
                "PRIORITY_RANK = {'High': 0, 'Medium': 1, 'Low': 2}\n"
                "\n"
                "@dataclass\n"
                "class Task:\n"
                "    title: str\n"
                "    description: str = ''\n"
                "    priority: str = 'Medium'\n"
                "    done: bool = False\n"
                "    id: str = field(default_factory=lambda: str(uuid.uuid4()))\n"
                "    created_at: str = field(default_factory=lambda: datetime.now().isoformat())\n"
                "\n"
                "st.title('Todo List')\n"
                "\n"
                "if 'tasks' not in st.session_state:\n"
                "    st.session_state.tasks = []\n"
                "\n"
                "# Sidebar filters\n"
                "status_filter = st.sidebar.selectbox('Filter', ['All', 'Active', 'Completed'])\n"
                "sort_by = st.sidebar.selectbox('Sort by', ['Priority', 'Date Created'])\n"
                "\n"
                "# Add task form\n"
                "with st.form('add_task'):\n"
                "    title = st.text_input('Task title')\n"
                "    desc = st.text_area('Description (optional)')\n"
                "    priority = st.selectbox('Priority', ['High', 'Medium', 'Low'])\n"
                "    if st.form_submit_button('Add Task') and title:\n"
                "        st.session_state.tasks.append(Task(title, desc, priority))\n"
                "        st.success(f'Added: {title}')\n"
                "\n"
                "# Filter and sort\n"
                "tasks = st.session_state.tasks\n"
                "if status_filter == 'Active':\n"
                "    tasks = [t for t in tasks if not t.done]\n"
                "elif status_filter == 'Completed':\n"
                "    tasks = [t for t in tasks if t.done]\n"
                "if sort_by == 'Priority':\n"
                "    tasks = sorted(tasks, key=lambda t: PRIORITY_RANK[t.priority])\n"
                "else:\n"
                "    tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)\n"
                "\n"
                "# Render tasks\n"
                "for task in tasks:\n"
                "    with st.expander(f\"{'✅' if task.done else '⬜'} {task.title} [{task.priority}]\"):\n"
                "        st.write(task.description or '_No description_')\n"
                "        st.caption(f'Created: {task.created_at}')\n"
                "        c1, c2 = st.columns(2)\n"
                "        if c1.button('Toggle Done', key=f'tog_{task.id}'):\n"
                "            task.done = not task.done\n"
                "            st.rerun()\n"
                "        if c2.button('Delete', key=f'del_{task.id}'):\n"
                "            st.session_state.tasks = [t for t in st.session_state.tasks if t.id != task.id]\n"
                "            st.rerun()\n"
                "\n"
                "# Footer summary\n"
                "total = len(st.session_state.tasks)\n"
                "done_count = sum(1 for t in st.session_state.tasks if t.done)\n"
                "st.caption(f'{done_count} of {total} tasks completed')"
            ),
        },
    },
    {
        "objective": "weather dashboard app",
        "outputs": {
            "description": (
                "A Streamlit-based weather dashboard application that accepts a city name "
                "and displays current weather conditions including temperature, humidity, "
                "wind speed, and a 5-day forecast. The app uses simulated weather data for "
                "demonstration purposes and presents information through cards and charts."
            ),
            "architecture": (
                "Single-file Streamlit application with a simulated data layer.\n"
                "Core components:\n"
                "1. Data Provider — Function that generates deterministic mock weather data "
                "based on city name hash (temperature, humidity, wind, conditions, 5-day "
                "forecast) so results are consistent per city.\n"
                "2. Current Weather Display — Metric cards using st.metric showing "
                "temperature, humidity, wind speed, and condition icon.\n"
                "3. Forecast Chart — st.line_chart displaying 5-day temperature trend.\n"
                "4. Search Module — Text input with search button and recent searches "
                "stored in session state.\n"
                "5. Theme — Uses st.set_page_config for wide layout and weather emoji "
                "mapping for visual polish."
            ),
            "ux_flow": (
                "1. App launches with title 'Weather Dashboard' in wide layout\n"
                "2. Top: text input for city name with 'Search' button and recent searches "
                "as quick-select chips\n"
                "3. On search: main area shows city name header with condition emoji\n"
                "4. Row of metric cards: Temperature (°C), Humidity (%), Wind (km/h), "
                "Condition\n"
                "5. Below metrics: 5-day forecast as a line chart with daily high/low\n"
                "6. Below chart: expandable table with detailed daily forecast data\n"
                "7. Sidebar: list of recent searches (last 5 cities) as clickable buttons\n"
                "8. Error state: friendly message if city name is empty"
            ),
            "code_flow": (
                "main() → st.set_page_config(layout='wide') → st.title() → "
                "init_recent_searches() → render_search_bar() → on_search(city) → "
                "generate_weather_data(city) → render_current_weather(data) → "
                "[st.metric(temp) | st.metric(humidity) | st.metric(wind)] → "
                "render_forecast_chart(data.forecast) → st.line_chart() → "
                "render_forecast_table(data.forecast) → st.dataframe() → "
                "update_recent_searches(city) → render_sidebar_recent()"
            ),
            "app_code": (
                "import streamlit as st\n"
                "import hashlib\n"
                "import random\n"
                "\n"
                "st.set_page_config(page_title='Weather Dashboard', layout='wide')\n"
                "st.title('🌤 Weather Dashboard')\n"
                "\n"
                "CONDITIONS = {\n"
                "    0: ('Sunny', '☀️'), 1: ('Cloudy', '☁️'), 2: ('Rainy', '🌧'),\n"
                "    3: ('Snowy', '❄️'), 4: ('Windy', '💨'),\n"
                "}\n"
                "\n"
                "def get_weather(city):\n"
                "    seed = int(hashlib.md5(city.lower().encode()).hexdigest(), 16) % 10000\n"
                "    rng = random.Random(seed)\n"
                "    cond_id = rng.randint(0, 4)\n"
                "    temp = rng.randint(-10, 40)\n"
                "    data = {\n"
                "        'temp': temp, 'humidity': rng.randint(20, 95),\n"
                "        'wind': rng.randint(0, 80),\n"
                "        'condition': CONDITIONS[cond_id][0],\n"
                "        'icon': CONDITIONS[cond_id][1],\n"
                "        'forecast': [{'day': f'Day {i+1}',\n"
                "                      'high': temp + rng.randint(-3, 5),\n"
                "                      'low': temp - rng.randint(2, 8)}\n"
                "                     for i in range(5)],\n"
                "    }\n"
                "    return data\n"
                "\n"
                "if 'recent' not in st.session_state:\n"
                "    st.session_state.recent = []\n"
                "\n"
                "# Search bar\n"
                "city = st.text_input('Enter city name', placeholder='e.g., Tokyo')\n"
                "\n"
                "# Sidebar recent searches\n"
                "st.sidebar.subheader('Recent Searches')\n"
                "for c in st.session_state.recent:\n"
                "    if st.sidebar.button(c, key=f'recent_{c}'):\n"
                "        city = c\n"
                "\n"
                "if city:\n"
                "    w = get_weather(city)\n"
                "    st.header(f\"{w['icon']} {city.title()} — {w['condition']}\")\n"
                "    c1, c2, c3 = st.columns(3)\n"
                "    c1.metric('Temperature', f\"{w['temp']}°C\")\n"
                "    c2.metric('Humidity', f\"{w['humidity']}%\")\n"
                "    c3.metric('Wind', f\"{w['wind']} km/h\")\n"
                "\n"
                "    import pandas as pd\n"
                "    df = pd.DataFrame(w['forecast'])\n"
                "    st.subheader('5-Day Forecast')\n"
                "    st.line_chart(df.set_index('day')[['high', 'low']])\n"
                "    with st.expander('Detailed Forecast'):\n"
                "        st.dataframe(df)\n"
                "\n"
                "    if city not in st.session_state.recent:\n"
                "        st.session_state.recent.insert(0, city)\n"
                "        st.session_state.recent = st.session_state.recent[:5]"
            ),
        },
    },
    {
        "objective": "quiz app",
        "outputs": {
            "description": (
                "A Streamlit-based multiple-choice quiz application with a built-in "
                "question bank, scoring system, and result summary. Users answer one "
                "question at a time, receive immediate feedback, and see a final score "
                "with correct/incorrect breakdown at the end."
            ),
            "architecture": (
                "Single-file Streamlit application with state-machine navigation.\n"
                "Core components:\n"
                "1. Question Bank — List of dicts, each containing question text, four "
                "options (A–D), correct answer key, and explanation.\n"
                "2. Quiz State Machine — Uses st.session_state to track: current question "
                "index, user answers list, score, and quiz phase (intro/quiz/result).\n"
                "3. Question Renderer — Displays question number, progress bar, question "
                "text, and radio buttons for options.\n"
                "4. Feedback Module — After submission shows correct/incorrect with "
                "explanation before advancing.\n"
                "5. Results Dashboard — Final screen showing total score, percentage, "
                "per-question breakdown with color coding."
            ),
            "ux_flow": (
                "1. App launches with title 'Quiz App' and a 'Start Quiz' button\n"
                "2. After start: progress bar at top shows question N of total\n"
                "3. Question text displayed prominently\n"
                "4. Four radio button options (A, B, C, D)\n"
                "5. 'Submit Answer' button — disabled until an option is selected\n"
                "6. After submit: green banner for correct, red for incorrect, plus "
                "explanation text\n"
                "7. 'Next Question' button appears to advance\n"
                "8. After last question: results screen with score fraction, percentage, "
                "emoji rating, and per-question table\n"
                "9. 'Restart Quiz' button resets all state"
            ),
            "code_flow": (
                "main() → st.title() → init_quiz_state() → "
                "if phase=='intro': render_intro() → on_start() → "
                "if phase=='quiz': render_question(idx) → st.radio(options) → "
                "on_submit() → check_answer(user_ans, correct_ans) → "
                "update_score() → show_feedback() → on_next() → "
                "if idx >= len(questions): set_phase('result') → "
                "if phase=='result': render_results(answers, score) → "
                "render_breakdown_table() → on_restart()"
            ),
            "app_code": (
                "import streamlit as st\n"
                "\n"
                "QUESTIONS = [\n"
                "    {'question': 'What keyword defines a function in Python?',\n"
                "     'options': ['func', 'define', 'def', 'function'],\n"
                "     'answer': 2, 'explanation': \"'def' is the keyword used to define functions.\"},\n"
                "    {'question': 'Which data structure uses key-value pairs?',\n"
                "     'options': ['List', 'Tuple', 'Set', 'Dictionary'],\n"
                "     'answer': 3, 'explanation': 'Dictionaries store key-value pairs.'},\n"
                "    {'question': 'What does len() return for an empty list?',\n"
                "     'options': ['None', '0', '-1', 'Error'],\n"
                "     'answer': 1, 'explanation': 'len([]) returns 0.'},\n"
                "    {'question': 'Which operator is used for exponentiation?',\n"
                "     'options': ['^', '**', 'exp', '//'],\n"
                "     'answer': 1, 'explanation': '** is the exponentiation operator.'},\n"
                "    {'question': 'What is the output of bool([])?',\n"
                "     'options': ['True', 'False', 'None', 'Error'],\n"
                "     'answer': 1, 'explanation': 'Empty collections are falsy in Python.'},\n"
                "]\n"
                "\n"
                "st.title('Quiz App')\n"
                "\n"
                "# Initialize state\n"
                "for key, val in [('phase', 'intro'), ('current_q', 0), ('score', 0),\n"
                "                 ('answers', []), ('submitted', False)]:\n"
                "    if key not in st.session_state:\n"
                "        st.session_state[key] = val\n"
                "\n"
                "if st.session_state.phase == 'intro':\n"
                "    st.write('Welcome! Test your Python knowledge.')\n"
                "    st.write(f'{len(QUESTIONS)} questions. Good luck!')\n"
                "    if st.button('Start Quiz'):\n"
                "        st.session_state.phase = 'quiz'\n"
                "        st.rerun()\n"
                "\n"
                "elif st.session_state.phase == 'quiz':\n"
                "    idx = st.session_state.current_q\n"
                "    q = QUESTIONS[idx]\n"
                "    st.progress((idx) / len(QUESTIONS))\n"
                "    st.subheader(f'Question {idx + 1} of {len(QUESTIONS)}')\n"
                "    st.write(q['question'])\n"
                "    choice = st.radio('Select your answer:', q['options'], key=f'q_{idx}')\n"
                "\n"
                "    if not st.session_state.submitted:\n"
                "        if st.button('Submit Answer'):\n"
                "            user_idx = q['options'].index(choice)\n"
                "            correct = user_idx == q['answer']\n"
                "            if correct:\n"
                "                st.session_state.score += 1\n"
                "            st.session_state.answers.append({\n"
                "                'question': q['question'], 'correct': correct,\n"
                "                'your_answer': choice,\n"
                "                'correct_answer': q['options'][q['answer']]})\n"
                "            st.session_state.submitted = True\n"
                "            st.rerun()\n"
                "    else:\n"
                "        last = st.session_state.answers[-1]\n"
                "        if last['correct']:\n"
                "            st.success(f\"Correct! {q['explanation']}\")\n"
                "        else:\n"
                "            st.error(f\"Wrong. The answer is {last['correct_answer']}. {q['explanation']}\")\n"
                "        if st.button('Next Question'):\n"
                "            st.session_state.current_q += 1\n"
                "            st.session_state.submitted = False\n"
                "            if st.session_state.current_q >= len(QUESTIONS):\n"
                "                st.session_state.phase = 'result'\n"
                "            st.rerun()\n"
                "\n"
                "elif st.session_state.phase == 'result':\n"
                "    score = st.session_state.score\n"
                "    total = len(QUESTIONS)\n"
                "    pct = int(score / total * 100)\n"
                "    emoji = '🏆' if pct >= 80 else '👍' if pct >= 50 else '📚'\n"
                "    st.header(f'{emoji} Your Score: {score}/{total} ({pct}%)')\n"
                "    for i, a in enumerate(st.session_state.answers):\n"
                "        icon = '✅' if a['correct'] else '❌'\n"
                "        st.write(f\"{icon} Q{i+1}: {a['question']}\")\n"
                "        if not a['correct']:\n"
                "            st.caption(f\"Your answer: {a['your_answer']} | Correct: {a['correct_answer']}\")\n"
                "    if st.button('Restart Quiz'):\n"
                "        for key in ['phase', 'current_q', 'score', 'answers', 'submitted']:\n"
                "            del st.session_state[key]\n"
                "        st.rerun()"
            ),
        },
    },
    {
        "objective": "note taking app",
        "outputs": {
            "description": (
                "A Streamlit-based note-taking application that allows users to create, "
                "edit, delete, and search notes. Each note has a title, body text, and "
                "timestamp. Notes persist in session state and support keyword search "
                "across titles and content."
            ),
            "architecture": (
                "Single-file Streamlit application with CRUD operations.\n"
                "Core components:\n"
                "1. Note Model — Dict with keys: id (uuid4), title, body, created_at, "
                "updated_at.\n"
                "2. Note Store — List of note dicts in st.session_state with helper "
                "functions for CRUD operations.\n"
                "3. Search Engine — Simple substring matching across title and body "
                "fields, case-insensitive.\n"
                "4. Editor View — Conditional rendering: 'create' mode shows empty form, "
                "'edit' mode pre-fills with existing note data.\n"
                "5. List View — Sidebar listing all notes with search box; main area "
                "shows selected note or editor."
            ),
            "ux_flow": (
                "1. App launches with title 'Note Taking App' and sidebar\n"
                "2. Sidebar top: search input field for filtering notes\n"
                "3. Sidebar: 'New Note' button, followed by list of note titles\n"
                "4. Clicking a note title shows it in the main area\n"
                "5. Main area (view mode): note title, body text, timestamps, 'Edit' "
                "and 'Delete' buttons\n"
                "6. 'Edit' switches to edit mode: editable title input and body textarea "
                "with 'Save' and 'Cancel' buttons\n"
                "7. 'New Note': switches to create mode with empty title and body\n"
                "8. 'Delete' removes the note and selects the next available note\n"
                "9. Search filters the sidebar list in real time as user types"
            ),
            "code_flow": (
                "main() → st.title() → init_notes() → render_sidebar() → "
                "search_input() → filter_notes(query) → render_note_list(filtered) → "
                "on_select_note(note_id) → if mode=='view': render_note_view(note) → "
                "[on_edit() | on_delete(note_id)] → "
                "if mode=='edit': render_note_editor(note) → on_save(note) → "
                "if mode=='create': render_note_editor(None) → on_create(title, body)"
            ),
            "app_code": (
                "import streamlit as st\n"
                "import uuid\n"
                "from datetime import datetime\n"
                "\n"
                "st.title('Note Taking App')\n"
                "\n"
                "# Initialize state\n"
                "if 'notes' not in st.session_state:\n"
                "    st.session_state.notes = [{\n"
                "        'id': str(uuid.uuid4()), 'title': 'Welcome',\n"
                "        'body': 'This is your first note. Edit or delete it, or create new ones!',\n"
                "        'created_at': datetime.now().isoformat(),\n"
                "        'updated_at': datetime.now().isoformat(),\n"
                "    }]\n"
                "if 'selected_id' not in st.session_state:\n"
                "    st.session_state.selected_id = st.session_state.notes[0]['id'] if st.session_state.notes else None\n"
                "if 'mode' not in st.session_state:\n"
                "    st.session_state.mode = 'view'\n"
                "\n"
                "def find_note(note_id):\n"
                "    return next((n for n in st.session_state.notes if n['id'] == note_id), None)\n"
                "\n"
                "# Sidebar\n"
                "search = st.sidebar.text_input('Search notes', placeholder='Type to search...')\n"
                "if st.sidebar.button('📝 New Note'):\n"
                "    st.session_state.mode = 'create'\n"
                "    st.session_state.selected_id = None\n"
                "\n"
                "filtered = [n for n in st.session_state.notes\n"
                "            if not search or search.lower() in n['title'].lower()\n"
                "            or search.lower() in n['body'].lower()]\n"
                "for note in filtered:\n"
                "    if st.sidebar.button(note['title'], key=f\"note_{note['id']}\"):\n"
                "        st.session_state.selected_id = note['id']\n"
                "        st.session_state.mode = 'view'\n"
                "        st.rerun()\n"
                "\n"
                "# Main area\n"
                "if st.session_state.mode == 'create':\n"
                "    st.subheader('Create New Note')\n"
                "    title = st.text_input('Title')\n"
                "    body = st.text_area('Content', height=300)\n"
                "    c1, c2 = st.columns(2)\n"
                "    if c1.button('Save') and title:\n"
                "        new_note = {'id': str(uuid.uuid4()), 'title': title, 'body': body,\n"
                "                    'created_at': datetime.now().isoformat(),\n"
                "                    'updated_at': datetime.now().isoformat()}\n"
                "        st.session_state.notes.insert(0, new_note)\n"
                "        st.session_state.selected_id = new_note['id']\n"
                "        st.session_state.mode = 'view'\n"
                "        st.rerun()\n"
                "    if c2.button('Cancel'):\n"
                "        st.session_state.mode = 'view'\n"
                "        st.rerun()\n"
                "\n"
                "elif st.session_state.selected_id:\n"
                "    note = find_note(st.session_state.selected_id)\n"
                "    if note:\n"
                "        if st.session_state.mode == 'view':\n"
                "            st.subheader(note['title'])\n"
                "            st.write(note['body'])\n"
                "            st.caption(f\"Created: {note['created_at']} | Updated: {note['updated_at']}\")\n"
                "            c1, c2 = st.columns(2)\n"
                "            if c1.button('Edit'):\n"
                "                st.session_state.mode = 'edit'\n"
                "                st.rerun()\n"
                "            if c2.button('Delete'):\n"
                "                st.session_state.notes = [n for n in st.session_state.notes if n['id'] != note['id']]\n"
                "                st.session_state.selected_id = st.session_state.notes[0]['id'] if st.session_state.notes else None\n"
                "                st.rerun()\n"
                "        elif st.session_state.mode == 'edit':\n"
                "            st.subheader('Edit Note')\n"
                "            title = st.text_input('Title', value=note['title'])\n"
                "            body = st.text_area('Content', value=note['body'], height=300)\n"
                "            c1, c2 = st.columns(2)\n"
                "            if c1.button('Save'):\n"
                "                note['title'] = title\n"
                "                note['body'] = body\n"
                "                note['updated_at'] = datetime.now().isoformat()\n"
                "                st.session_state.mode = 'view'\n"
                "                st.rerun()\n"
                "            if c2.button('Cancel'):\n"
                "                st.session_state.mode = 'view'\n"
                "                st.rerun()\n"
                "\n"
                "else:\n"
                "    st.info('Select a note from the sidebar or create a new one.')"
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

# Base: 2025-03-01 00:00:00 UTC in microseconds
BASE_TIMESTAMP_US = 1_740_787_200_000_000

# Cumulative wall-clock latency per step (µs) — realistic GPT-4 API durations
STEP_DELAYS_US = [3_200_000, 8_100_000, 15_400_000, 19_700_000, 41_000_000]

# Gap between consecutive sessions (~90 seconds in µs)
SESSION_GAP_US = 90_000_000


def build_input(sys_instructions: str, user_task: str) -> str:
    """Combine system and user messages into a single flat input string."""
    system_msg = SYSTEM_PROMPT_HEADER.format(instructions=sys_instructions)
    return f"[System]\n{system_msg}\n\n[User]\n{user_task}"


def generate_session_trace(session_idx: int, session: dict) -> list:
    """Generate 5 trace records for one session (one per pipeline step)."""
    objective = session["objective"]
    outputs = session["outputs"]
    session_id = hashlib.md5(objective.encode("utf-8")).hexdigest()

    session_start_us = BASE_TIMESTAMP_US + session_idx * SESSION_GAP_US
    records = []
    accumulated: dict = {}

    for step_idx, step_name in enumerate(STEP_NAMES):
        step = STEPS_CONFIG[step_name]
        fmt_kwargs = {"objective": objective, **accumulated}
        sys_instructions = step["sys_instructions"].format(**fmt_kwargs)
        user_task = step["user_task"].format(**fmt_kwargs)

        records.append({
            "timestamp": session_start_us + STEP_DELAYS_US[step_idx],
            "input": build_input(sys_instructions, user_task),
            "output": outputs[step_name],
            "session_id": session_id,
        })
        accumulated[step_name] = outputs[step_name]

    return records


def main():
    parser = argparse.ArgumentParser(description="Generate PyCodeAGI synthetic trace")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (default: trace.jsonl in script dir)")
    args = parser.parse_args()

    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "trace.jsonl")

    all_records = []
    for session_idx, session in enumerate(SYNTHETIC_SESSIONS):
        records = generate_session_trace(session_idx, session)
        all_records.extend(records)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_records)} trace entries "
          f"({len(SYNTHETIC_SESSIONS)} sessions × {len(STEP_NAMES)} steps)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
