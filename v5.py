import os
import json
import tkinter as tk
from tkinter import messagebox, font
import threading
import re
from collections import deque
from functools import lru_cache
from langdetect import detect, LangDetectException
import pyttsx4
import language_tool_python

# Constants
REGEX_RULES_FILE = "regex_rules11.json"
DATASET_A_FILE = "lang8_corrected_pairs.json"
DATASET_B_FILE = "text_correction_dataset.json"
SUPPORTED_LANGUAGE = "en"
BART_MODEL_NAME = "facebook/bart-large-cnn"

# Initialize LanguageTool
GRAMMAR_TOOL = language_tool_python.LanguageTool('en-US')

# Pre-populated datasets
INITIAL_DATASET_A = {
    "the president was standing in the front row and the every female enployees were surrounding him.": "the president was standing in the front row and all the female employees were surrounding him.",
    "in us, i heard that arrested people will not lose their jobs.": "in the us, i heard that people arrested will not lose their jobs.",
    "i always practice pronounce, but getting more difficult.": "i always practise pronunciation, but it is getting more difficult.",
    "if i said that sentence, people will be heard same pronounciation.": "if i said that sentence, people would hear the same pronunciation.",
    "i just learnt alphabet.": "i have just learnt the alphabet.",
    "i'm always not patient enough to take a self - study for a long time.": "i'm never patient enough to take a self - study course for a long time.",
    "but it is also said that it is very difficult to learn well both of them.": "but it is also said that it is very difficult to learn both of them well.",
    "then i asked my brother for an advice, he said that  i like japanese.": "then i asked my brother for some advice. he said that he liked japanese.",
    "i was acquainted with him in a chinese's room chat when i was at 9th grade about 4 years ago.": "i was acquainted with him in a chinese chat room when i was in 9th grade, about 4 years ago.",
    "he is very humorious and handsome > _ <.": "he is very handsome and humorous > _ <.",
    "that is the reason why i did not choose english department despite i like it best.": "that is the reason why i did not choose the english department despite the fact that i like it best.",
    "this time we still have to test 4 japanese's skill.": "this time we still had to test 4 japanese people's skills.",
    "i'm very nervous of speaking and listening skill.": "i'm very nervous about my speaking and listening skills.",
    "i'll have to write anything in japanese as topic's request.": "i'll have to write something in japanese as per the topic's request.",
    "( my computer's cd room was broken so that i cannot install japanese font.": "( my computer's cd - rom was broken so that i could not install a japanese font.",
    "however they are unsuccessful because most of their disarmament were not achieve.": "however, they were unsuccessful because most of their disarmament goals were not achieved.",
    "recently, i'm immerced in running for my weight control.": "recently, i've been immersed in running for my weight control.",
    "but, my weight is more bigger than two week ago.": "but, my weight is more than two weeks ago.",
    "i have given up smoking for two week because i want to get my health.": "i have given up smoking for two weeks because i want to improve my health.",
}

INITIAL_DATASET_B = {
    "he go to school yesterday": "he went to school yesterday",
    "she dont like apples": "she doesn't like apples",
}

def load_dataset(file_path, default_data):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                return json.load(file)
        except UnicodeDecodeError:
            with open(file_path, "r", encoding='latin-1') as file:
                return json.load(file)
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(default_data, file, indent=4, ensure_ascii=False)
    return default_data.copy()

def save_dataset(file_path, data):
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def add_to_dataset_b(input_text, corrected_text):
    dataset_b = load_dataset(DATASET_B_FILE, INITIAL_DATASET_B)
    dataset_b[input_text] = corrected_text
    save_dataset(DATASET_B_FILE, dataset_b)

def get_corrected_text_from_datasets(input_text):
    dataset_a = load_dataset(DATASET_A_FILE, INITIAL_DATASET_A)
    dataset_b = load_dataset(DATASET_B_FILE, INITIAL_DATASET_B)
    
    if input_text in dataset_a:
        return dataset_a[input_text], "dataset_a"
    if input_text in dataset_b:
        return dataset_b[input_text], "dataset_b"
    return None, None

def is_probably_english(sentence):
    common_words = {"the", "and", "to", "of", "in", "a", "i", "is", "it", "that"}
    words = set(sentence.lower().split())
    return len(words.intersection(common_words)) >= 2

def load_regex_rules():
    try:
        with open(REGEX_RULES_FILE, "r", encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        rules = [
            {"pattern": r"\b(a|A)(\s+)([aeiouAEIOU]\w*)", "replacement": lambda m: f"an{m.group(2)}{m.group(3)}" if m.group(1).islower() else f"An{m.group(2)}{m.group(3)}"},
            {"pattern": r"\b(screenested in)\s+learn\b", "replacement": r"\1 learning"},
            {"pattern": r"\b(interested in)\s+learn\b", "replacement": r"\1 learning"},
            {"pattern": r"(\b[Ii]\s+went\b.*?\band\s+[Ii]\s+)buy\b", "replacement": r"\1bought"},
            {"pattern": r"\beach\s+(\w+)\s+have\b", "replacement": r"each \1 has"},
            {"pattern": r"\b(he|she)\s+to\s+go\s+(\w+)\b", "replacement": r"\1 has to go to \2"},
            {"pattern": r"\b(the\s+\w+s\s+in\s+the\s+\w+)\s+is\b", "replacement": r"\1 are"},
            {"pattern": r"\bas her\b", "replacement": "as she"},
            {"pattern": r"\bas him\b", "replacement": "as he"},
            {"pattern": r"\bas them\b", "replacement": "as they"},
            {"pattern": r"\bIf I were her\b", "replacement": "If I were she"},
            {"pattern": r"\bIf I were him\b", "replacement": "If I were he"},
            {"pattern": r"\bIf I were them\b", "replacement": "If I were they"},
            {"pattern": r"\bmany\s+a\s+(\w+)\s+have\b", "replacement": r"many a \1 has"},
            {"pattern": r"^(Hardly|Scarcely)\s+I\s+", "replacement": r"\1 had I "},
            {"pattern": r"\b(A\s+(?:large\s+)?number\s+of\s+\w+)\s+was\b", "replacement": r"\1 were"},
            {"pattern": r"\b(The\s+number\s+of\s+\w+)\s+have\b", "replacement": r"\1 has"},
            {"pattern": r"\bIt's\s+(high|about)\s+time\s+you\s+(\w+)\b", "replacement": r"It's \1 time you \2ed"},
            {"pattern": r"\b(I|He|She|They)\s+(enjoyed|hurt|cheated|applied)\b", "replacement": r"\1 \2 myself"},
            {"pattern": r"^Being\s+([a-zA-Z]+\s+day)", "replacement": r"It being \1"},
            {"pattern": r"\blest\s+we\s+shall\b", "replacement": "lest we should"},
            {"pattern": r"\bthan\s+any\s+(\w+)\b", "replacement": r"than any other \1"},
            {"pattern": r"(\w+)\s+is\s+(\w+er)\s+than\s+(\w+)", "replacement": r"\1 is more \2 than \3"},
            {"pattern": r"\b(You),\s*(he|He),\s*and\s*(I)\b", "replacement": "I, You and He"},
            {"pattern": r"\b(the\s+\w+)\s+run\b", "replacement": r"\1 runs"},
            {"pattern": r"\b(an|An)\s+([^aeiouAEIOU]\w*)", "replacement": lambda m: f"a {m.group(2)}" if m.group(1).islower() else f"A {m.group(2)}"},
            {"pattern": r"\b(an|An)\s+(?!(hour|honest|honor|heir|herb|hospital|hotel|historical|horror|habit|harmony|homage|humble|union|united|universe|university|unicorn|unique|user|usual|utopia|ubiquitous|one|once|one[\\s-]time|one[\\s-]off|European|eulogy|eureka|ewe|ewer)\\b)([^aeiouAEIOU]\\w*)", "replacement": lambda m: f"a {m.group(3)}" if m.group(1).islower() else f"A {m.group(3)}"}
        ]
        return rules

@lru_cache(maxsize=1)
def get_bart_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME)
    return tokenizer, model

def apply_regex_rules(sentence, regex_rules):
    original = sentence
    for rule in regex_rules:
        try:
            sentence = re.sub(rule["pattern"], rule["replacement"], sentence, flags=re.IGNORECASE)
        except re.error as e:
            print(f"Regex error in rule {rule}: {e}")
    return sentence, original != sentence

def grammar_check_with_languagetool(sentence):
    matches = GRAMMAR_TOOL.check(sentence)
    suggestions = []
    corrected = language_tool_python.utils.correct(sentence, matches)
    
    for match in matches:
        suggestions.append({
            'error': match.context,
            'message': match.message,
            'replacement': match.replacements[0] if match.replacements else '',
            'offset': match.offset
        })
    
    return corrected, suggestions, len(matches) > 0

def refine_with_bart(tokenizer, model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
    refined = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    if len(refined.split()) > len(sentence.split()) * 1.2:
        return sentence, False
    return refined, refined != sentence

def correct_sentence_structure(sentence):
    corrected_text, source = get_corrected_text_from_datasets(sentence)
    if corrected_text:
        return {
            "original": sentence,
            "corrected": corrected_text,
            "errors": [],
            "source": source,
            "changes_made": sentence != corrected_text,
            "stages": [("Dataset", corrected_text)] if sentence != corrected_text else []
        }
    
    warning = None
    try:
        detected_lang = detect(sentence)
        if detected_lang != SUPPORTED_LANGUAGE and not is_probably_english(sentence):
            warning = f"Detected language is not English ({detected_lang}). Results may be inaccurate."
    except LangDetectException:
        warning = "Could not detect language. Results may be inaccurate."
    
    stages = []
    suggestions = []
    changes_made = False
    current_text = sentence

    # Step 1: Apply regex rules
    regex_rules = load_regex_rules()
    regex_corrected, regex_changed = apply_regex_rules(current_text, regex_rules)
    if regex_changed:
        stages.append(("Regex", regex_corrected))
        changes_made = True
        current_text = regex_corrected

    # Step 2: Grammar Check with LanguageTool
    grammar_corrected, grammar_suggestions, grammar_changed = grammar_check_with_languagetool(current_text)
    if grammar_changed:
        stages.append(("Grammar Tool", grammar_corrected))
        suggestions.extend(grammar_suggestions)
        changes_made = True
        current_text = grammar_corrected

    # Step 3: Refine with BART (always applied)
    tokenizer, model = get_bart_model()
    bart_corrected, bart_changed = refine_with_bart(tokenizer, model, current_text)
    if bart_changed:
        stages.append(("BART Refinement", bart_corrected))
        changes_made = True
        current_text = bart_corrected
    else:
        stages.append(("BART Refinement", bart_corrected))  # Log even if no change

    if changes_made:
        add_to_dataset_b(sentence, current_text)

    return {
        "warning": warning,
        "original": sentence,
        "corrected": current_text,
        "errors": suggestions,
        "source": "new_correction",
        "changes_made": changes_made,
        "stages": stages
    }

class EGECGrammarCorrectionBot:
    def __init__(self, root):
        self.root = root
        self.root.title("EGEC Grammar Correction Bot 📝")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.BACKGROUND_COLOR = "#2E3440"
        self.FOREGROUND_COLOR = "#D8DEE9"
        self.ACCENT_COLOR = "#5E81AC"
        self.ERROR_COLOR = "#BF616A"
        self.SUCCESS_COLOR = "#A3BE8C"
        self.BUTTON_COLOR = "#4C566A"
        self.BUTTON_HOVER_COLOR = "#434C5E"
        self.TEXT_COLOR = "#ECEFF4"

        self.root.configure(bg=self.BACKGROUND_COLOR)
        self.root.geometry("800x600")

        self.DEFAULT_FONT = font.nametofont("TkDefaultFont")
        self.DEFAULT_FONT.configure(size=12)
        self.LARGE_FONT = ("Helvetica", 14, "bold")
        self.TEXT_FONT = ("Helvetica", 12)

        self.processing_button = None
        self.blink_state = False

        self.speech_queue = deque()
        self.is_speaking = False
        self.engine = pyttsx4.init()
        self.engine.setProperty('rate', self.engine.getProperty('rate') - 50)
        self.engine.setProperty('voice', self.engine.getProperty('voices')[0].id)
        self.engine.connect('started-utterance', self.on_speech_start)
        self.engine.connect('finished-utterance', self.on_speech_end)

        self.create_widgets()

    def create_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        main_frame = tk.Frame(self.root, bg=self.BACKGROUND_COLOR)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        header = tk.Label(main_frame, text="✨ EGEC Grammar Correction Bot ✨", bg=self.BACKGROUND_COLOR, fg=self.ACCENT_COLOR, font=("Helvetica", 16, "bold"))
        header.pack(pady=5)

        input_frame = tk.Frame(main_frame, bg=self.BACKGROUND_COLOR)
        input_frame.pack(fill="x", pady=(0,10))
        
        tk.Label(input_frame, text="📜 Enter text or use voice input:", bg=self.BACKGROUND_COLOR, fg=self.FOREGROUND_COLOR, font=self.LARGE_FONT).pack(anchor="w")
        
        self.text_input = tk.Entry(input_frame, width=50, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR, insertbackground=self.TEXT_COLOR, font=self.TEXT_FONT, relief=tk.FLAT, borderwidth=2)
        self.text_input.pack(fill="x", pady=5)

        button_frame = tk.Frame(main_frame, bg=self.BACKGROUND_COLOR)
        button_frame.pack(fill="x", pady=5)

        self.voice_button = tk.Button(button_frame, text="🎤 Use Voice Input", command=self.recognize_speech, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR, activebackground=self.BUTTON_HOVER_COLOR, activeforeground=self.TEXT_COLOR, relief=tk.FLAT, font=self.LARGE_FONT, padx=10, pady=5)
        self.voice_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.speak_button = tk.Button(button_frame, text="🔊 Speak Text", command=self.convert_text_to_voice, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR, activebackground=self.BUTTON_HOVER_COLOR, activeforeground=self.TEXT_COLOR, relief=tk.FLAT, font=self.LARGE_FONT, padx=10, pady=5)
        self.speak_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.correct_button = tk.Button(button_frame, text="✅ Correct Text", command=self.correct_text, bg=self.BUTTON_COLOR, fg=self.TEXT_COLOR, activebackground=self.BUTTON_HOVER_COLOR, activeforeground=self.TEXT_COLOR, relief=tk.FLAT, font=self.LARGE_FONT, padx=10, pady=5)
        self.correct_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)

        output_frame = tk.Frame(main_frame, bg=self.BACKGROUND_COLOR)
        output_frame.pack(fill="both", expand=True)

        tk.Label(output_frame, text="📋 Original Text:", bg=self.BACKGROUND_COLOR, fg=self.FOREGROUND_COLOR, font=self.LARGE_FONT).pack(anchor="w")
        self.original_text = tk.Text(output_frame, height=3, width=60, bg="#333333", fg=self.TEXT_COLOR, font=self.DEFAULT_FONT, wrap=tk.WORD, relief=tk.FLAT, borderwidth=2)
        self.original_text.pack(fill="x", pady=(0,10))

        tk.Label(output_frame, text="✅ Corrected Text:", bg=self.BACKGROUND_COLOR, fg=self.SUCCESS_COLOR, font=self.LARGE_FONT).pack(anchor="w")
        self.corrected_text = tk.Text(output_frame, height=3, width=60, bg="#2b2b2b", fg=self.SUCCESS_COLOR, font=self.DEFAULT_FONT, wrap=tk.WORD, relief=tk.FLAT, borderwidth=2)
        self.corrected_text.pack(fill="x", pady=(0,10))

        tk.Label(output_frame, text="💡 Suggestions:", bg=self.BACKGROUND_COLOR, fg=self.ACCENT_COLOR, font=self.LARGE_FONT).pack(anchor="w")
        self.suggestions_text = tk.Text(output_frame, height=5, width=60, bg="#2b2b2b", fg=self.ACCENT_COLOR, font=self.DEFAULT_FONT, wrap=tk.WORD, relief=tk.FLAT, borderwidth=2)
        self.suggestions_text.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(main_frame, command=self.suggestions_text.yview, bg=self.BACKGROUND_COLOR)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.suggestions_text.config(yscrollcommand=scrollbar.set)

    def start_processing(self, button):
        self.processing_button = button
        self.disable_buttons()
        self.update_button_colors()
        self.start_blinking()

    def stop_processing(self):
        self.processing_button = None
        self.stop_blinking()
        self.enable_buttons()
        self.update_button_colors()

    def update_button_colors(self):
        for button in [self.voice_button, self.speak_button, self.correct_button]:
            if button == self.processing_button:
                button.config(bg=self.ACCENT_COLOR)
            elif self.processing_button is not None:
                button.config(bg=self.BUTTON_HOVER_COLOR)
            else:
                button.config(bg=self.BUTTON_COLOR)

    def start_blinking(self):
        if self.processing_button:
            self.blink_state = not self.blink_state
            color = self.ACCENT_COLOR if self.blink_state else self.BUTTON_HOVER_COLOR
            self.processing_button.config(bg=color)
            self.root.after(500, self.start_blinking)

    def stop_blinking(self):
        self.blink_state = False
        if self.processing_button:
            self.processing_button.config(bg=self.BUTTON_COLOR)

    def disable_buttons(self):
        for button in [self.voice_button, self.speak_button, self.correct_button]:
            button.config(state=tk.DISABLED)

    def enable_buttons(self):
        for button in [self.voice_button, self.speak_button, self.correct_button]:
            button.config(state=tk.NORMAL)

    def convert_text_to_voice(self):
        text = self.text_input.get()
        if not text.strip():
            messagebox.showerror("Error", "Please enter some text!")
            return
        
        self.start_processing(self.speak_button)
        threading.Thread(target=self._speak_text_thread, args=(text,), daemon=True).start()

    def _speak_text_thread(self, text):
        self.speak_text(text, interrupt=True)
        self.root.after(0, self.stop_processing)

    def correct_text(self):
        text = self.text_input.get()
        if not text.strip():
            messagebox.showerror("Error", "Please enter some text to correct!")
            return
        
        self.start_processing(self.correct_button)
        threading.Thread(target=self._correct_text_thread, args=(text,), daemon=True).start()

    def _correct_text_thread(self, text):
        result = correct_sentence_structure(text)
        
        self.root.after(0, lambda: self.original_text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.original_text.insert(tk.END, f"{result['original']}\n"))
        
        self.root.after(0, lambda: self.corrected_text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.corrected_text.insert(tk.END, f"{result['corrected']}\n"))
        
        self.root.after(0, lambda: self.suggestions_text.delete(1.0, tk.END))
        if result.get("warning"):
            self.root.after(0, lambda: messagebox.showwarning("Warning", result["warning"]))
        
        speech_output = f"Original text: {result['original']}\nCorrected text: {result['corrected']}"
        suggestions_text = ""
        
        if result["source"] in ["dataset_a", "dataset_b"]:
            suggestions_text += f"📚 (Result loaded from {result['source'].replace('_', ' ')})\n"
            if result["changes_made"]:
                suggestions_text += f"✅ Change: '{result['original']}' → '{result['corrected']}'\n"
                speech_output += f"\nChange made: '{result['original']}' changed to '{result['corrected']}'"
            else:
                suggestions_text += "✔️ No corrections needed.\n"
                speech_output += "\nNo corrections needed."
        else:
            if result["stages"]:
                suggestions_text += "🔍 Correction stages:\n"
                prev_text = result["original"]
                for stage, corrected in result["stages"]:
                    if prev_text != corrected:
                        suggestions_text += f"{stage}: '{prev_text}' → '{corrected}'\n"
                        speech_output += f"\n{stage} changed '{prev_text}' to '{corrected}'"
                    else:
                        suggestions_text += f"{stage}: No change ('{prev_text}')\n"
                    prev_text = corrected
            
            if result.get("errors"):
                suggestions_text += "\n🔧 Grammar Issues Found:\n"
                for error in result["errors"]:
                    suggestions_text += (f"• At position {error['offset']}: {error['message']}\n"
                                        f"  Example: {error['error']}\n"
                                        f"  Suggested: {error['replacement'] or 'N/A'}\n\n")
                speech_output += "\nGrammar issues were addressed."

        self.root.after(0, lambda: self.suggestions_text.insert(tk.END, suggestions_text))
        self.root.after(0, lambda: self.suggestions_text.see(tk.END))

        self.root.after(0, lambda: self.root.update_idletasks())
        self.speak_text(speech_output, interrupt=True, display=False)
        self.root.after(0, self.stop_processing)

    def recognize_speech(self):
        self.start_processing(self.voice_button)
        
        self.suggestions_text.insert(tk.END, "🎙️ Listening...\n")
        
        def recognition_task():
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio)
                    self.root.after(0, lambda: self.text_input.delete(0, tk.END))
                    self.root.after(0, lambda: self.text_input.insert(0, text))
                    self.root.after(0, lambda: self.suggestions_text.insert(tk.END, f"🎤 Recognized: {text}\n"))
                    self.root.after(0, lambda: self.speak_text(text, interrupt=True))
            except Exception as ex:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Speech recognition failed: {str(ex)}"))
            finally:
                self.root.after(0, self.stop_processing)
        
        threading.Thread(target=recognition_task, daemon=True).start()

    def speak_text(self, text, interrupt=False, display=True):
        if interrupt and self.is_speaking:
            self.engine.stop()
            self.speech_queue.clear()
        
        if display:
            self.suggestions_text.insert(tk.END, f"🔊 Speaking: {text}\n")
            self.suggestions_text.see(tk.END)
            self.root.update_idletasks()

        self.engine.say(text)
        self.engine.runAndWait()

    def on_speech_start(self, name):
        self.is_speaking = True
        self.root.config(cursor="watch")

    def on_speech_end(self, name, completed):
        self.root.config(cursor="")
        self.is_speaking = False
        if self.speech_queue:
            next_text = self.speech_queue.popleft()
            self.speak_text(next_text, display=False)

    def on_close(self):
        self.engine.stop()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = EGECGrammarCorrectionBot(root)
    root.mainloop()