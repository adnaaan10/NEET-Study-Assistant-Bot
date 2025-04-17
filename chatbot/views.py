from django.shortcuts import render, redirect
from utils.rag_pipeline import ask_question, is_mcq_request, extract_mcq_parameters
import os
import re

# Create a unique identifier when server starts
SERVER_RUN_ID = os.urandom(16).hex()


def parse_mcq_response(response_text):
    """Parse the MCQ response into a structured format for the template"""
    questions = []
    current_question = None

    # Split into potential question blocks
    blocks = re.split(r'\n\s*\n', response_text.strip())

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue

        # Check if block starts with question
        if re.match(r'^(Q?\d+\.|\\d+\.)', lines[0]):
            if current_question:
                questions.append(current_question)
            current_question = {
                'text': re.sub(r'^(Q?\d+\.)\s*', '', lines[0]),
                'options': []
            }
            lines = lines[1:]

        # Process options
        for line in lines:
            if re.match(r'^[a-d]\)', line):
                option_text = re.sub(r'^[a-d]\)\s*', '', line)
                is_correct = '(Correct)' in option_text
                option_text = option_text.replace('(Correct)', '').strip()
                if current_question:
                    current_question['options'].append({
                        'letter': line[0],
                        'text': option_text,
                        'is_correct': is_correct
                    })

    if current_question and current_question['options']:
        questions.append(current_question)

    return questions


def extract_mcq_header(response_text):
    """Extract the header line from MCQ response"""
    lines = response_text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if clean_line and not re.match(r'^(Q?\d+\.|[a-d]\)|\\d+\.)', clean_line):
            return clean_line
    return "NEET Practice Questions"


def index(request):
    # Clear history if server has restarted
    if request.session.get('server_run_id') != SERVER_RUN_ID:
        request.session['chat_history'] = []
        request.session['server_run_id'] = SERVER_RUN_ID
        request.session.modified = True

    # Handle Clear button
    if request.method == "POST" and "clear" in request.POST:
        request.session["chat_history"] = []
        request.session.modified = True
        return redirect("index")

    # Process queries
    if request.method == "POST" and "query" in request.POST:
        query = request.POST.get("query", "").strip()
        if query:
            chat_history = request.session.get("chat_history", [])

            try:
                response = ask_question(query)
                is_mcq = is_mcq_request(query)
                mcq_header = ""
                mcq_questions = []

                if is_mcq:
                    mcq_header = extract_mcq_header(response)
                    mcq_questions = parse_mcq_response(response)
                    # Validate parsed questions
                    if not mcq_questions or any(len(q['options']) != 4 for q in mcq_questions):
                        is_mcq = False  # Fallback to regular message
                        response = "Couldn't generate valid MCQs. Please try rephrasing your request."

                chat_history.append({
                    "user": query,
                    "bot": response,
                    "is_mcq": is_mcq,
                    "mcq_header": mcq_header,
                    "mcq_questions": mcq_questions
                })

                request.session["chat_history"] = chat_history
                request.session.modified = True

            except Exception as e:
                # Handle unexpected errors
                error_msg = "An error occurred. Please try again."
                chat_history.append({
                    "user": query,
                    "bot": error_msg,
                    "is_mcq": False
                })
                request.session.modified = True

    return render(request, "chatbot/chatbot.html", {
        "chat_history": request.session.get("chat_history", [])
    })