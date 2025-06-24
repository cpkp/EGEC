import json
import tkinter as tk
from tkinter import filedialog
from transformers import pipeline

class GrammarCorrectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grammar Correction Refiner")
        
        # GUI Components
        self.load_btn = tk.Button(root, text="Load JSON File", command=self.load_json)
        self.load_btn.pack(pady=10)
        
        self.refine_btn = tk.Button(root, text="Refine", command=self.refine_data, state=tk.DISABLED)
        self.refine_btn.pack(pady=5)
        
        self.status_label = tk.Label(root, text="", wraplength=400)
        self.status_label.pack(pady=10)
        
        # Initialize variables
        self.data = None
        self.file_path = ""
        self.corrector = None

    def load_json(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if self.file_path:
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
                self.refine_btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Loaded: {self.file_path}")
            except Exception as e:
                self.status_label.config(text=f"Error loading file: {str(e)}")

    def refine_data(self):
        if not self.data:
            return

        try:
            self.status_label.config(text="Initializing Grammar Correction model...")
            self.root.update()
            
            # Initialize T5 grammar correction pipeline
            self.corrector = pipeline(
                'text2text-generation',
                model='vennify/t5-base-grammar-correction',
                max_length=128
            )
            
            corrected_count = 0
            total_entries = len(self.data)
            
            for idx, (original, corrected) in enumerate(self.data.copy().items()):
                # Update status
                self.status_label.config(text=f"Processing {idx+1}/{total_entries}: {original}")
                self.root.update()
                
                # Only process entries needing correction (original == corrected)
                if original == corrected:
                    # Generate correction using T5 model
                    result = self.corrector(
                        f"grammar: {original}",  # Special prompt format the model expects
                        max_length=128,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    # Extract corrected sentence
                    new_corrected = result[0]['generated_text'].strip()
                    
                    # Update data
                    self.data[original] = new_corrected
                    corrected_count += 1
            
            # Save refined data
            save_path = self.file_path.replace('.json', '_refined.json')
            with open(save_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            
            self.status_label.config(text=f"Refined {corrected_count} entries. Saved to:\n{save_path}")
            self.refine_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            self.status_label.config(text=f"Error during processing: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GrammarCorrectorApp(root)
    root.mainloop()