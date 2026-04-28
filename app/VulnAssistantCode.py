import os
import io
import zipfile
import logging
from typing import List, Optional

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Environment & Logging
# ----------------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("pentest-assistant")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Provide it via environment or .env file.")

# Configure OpenAI client with reasonable timeouts and retries
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=2)


# ----------------------------
# Core Assistant
# ----------------------------
class PenTestAssistant:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.mode = "professional"
        self.stage: Optional[str] = None
        self.authorized: bool = False  # gate potentially sensitive outputs
        self.conversation = [
            {
                "role": "system",
                "content": (
                    "You are a professional AI Penetration Testing Assistant. "
                    "Target users are cybersecurity professionals, beginners in penetration testing and students. "
                    "Your responsibilities include:\n"
                    "1. Guiding the user step-by-step through penetration testing processes.\n"
                    "2. Analyzing scans, technical data and logs.\n"
                    "3. Point out every found vulnerability and prioritize them by severity.\n"
                    "4. Give safe, high-level guidance on exploitation theory strictly for authorized, ethical pentesting contexts.\n"
                    "5. Help when the user gets stuck or has questions.\n"
                    "6. Generate clear, professional final reports with findings, severity and recommendations.\n"
                    "7. Explain potential impact in an educational way.\n"
                    "8. Provide mitigation techniques and recommendations for non-technical stakeholders.\n"
                    "Never assist with illegal or unauthorized activity. If the user is not authorized, refuse."
                ),
            }
        ]

    def set_mode(self, mode: str):
        self.mode = mode

    def set_stage(self, stage: Optional[str]):
        self.stage = stage

    def set_authorized(self, authorized: bool):
        self.authorized = bool(authorized)

    def _ensure_authorized(self):
        if not self.authorized:
            return (
                "Authorization required: Please confirm you have explicit permission to test the target(s). "
                "I can still help with defensive/security best practices in the meantime."
            )
        return None

    def _response(self, temperature=0.4, max_tokens=1200):
        resp = client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def chat(self, user_input: str) -> str:
        role_prompt = ""
        if self.mode == "beginner":
            role_prompt = "Explain findings simply with analogies where helpful."
        elif self.mode == "professional":
            role_prompt = "Provide concise, technical, prioritized answers."
        if self.stage:
            role_prompt += f" Focus especially on this phase: {self.stage}."

        unauthorized_msg = self._ensure_authorized()
        if unauthorized_msg:
            self.conversation.append({"role": "user", "content": f"GENERAL_ADVICE_ONLY\n\n{user_input}"})
            message = self._response(temperature=0.5, max_tokens=800)
            return unauthorized_msg + "\n\n—\nGeneral guidance:\n" + message

        self.conversation.append({"role": "user", "content": f"{role_prompt}\n\n{user_input}".strip()})
        message = self._response(temperature=0.4, max_tokens=1200)
        self.conversation.append({"role": "assistant", "content": message})
        return message

    def analyze_scans(self, files=None, pasted_text: str = "") -> str:
        # Authorization check
        unauthorized_msg = self._ensure_authorized()
        if unauthorized_msg:
            return unauthorized_msg

        texts: List[str] = []

        def _read_file_string(path: str) -> Optional[str]:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    return fh.read()
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
                return f"[!] Failed to read {path}: {e}"

        if files:
            if not isinstance(files, list):
                files = [files]
            for f in files:
                path = None
                if isinstance(f, str):
                    path = f
                elif hasattr(f, "name") and isinstance(f.name, str):
                    path = f.name
                elif hasattr(f, "path") and isinstance(f.path, str):
                    path = f.path

                if not path:
                    continue

                lower = path.lower()
                try:
                    if lower.endswith(".zip"):
                        with zipfile.ZipFile(path, "r") as zf:
                            for zi in zf.infolist():
                                if zi.is_dir():
                                    continue
                                name_low = zi.filename.lower()
                                if not name_low.endswith((".txt", ".log", ".xml")):
                                    continue
                                # Guardrail: limit per-file size to 3 MB
                                if zi.file_size > 3 * 1024 * 1024:
                                    texts.append(f"[!] Skipped {zi.filename}: file too large (>3 MB).")
                                    continue
                                with zf.open(zi, "r") as fh:
                                    try:
                                        texts.append(fh.read().decode("utf-8", errors="ignore"))
                                    except Exception as e:
                                        texts.append(f"[!] Failed to read {zi.filename}: {e}")
                    else:
                        texts.append(_read_file_string(path) or "")
                except Exception as e:
                    texts.append(f"[!] Failed to process {path}: {e}")

        if isinstance(pasted_text, str) and pasted_text.strip():
            texts.append(pasted_text.strip())

        combined = "\n\n".join(x for x in texts if x).strip()
        if not combined:
            return "No data to analyze: add files or paste scan results."

        self.conversation.append({
            "role": "user",
            "content": (
                "Analyze the following scan results and create a markdown table. "
                "The table must be sorted from most severe issues at the top to least severe at the bottom. "
                "Columns: | Vulnerability | Severity | Impact | Ethical Exploitation Guidance (high-level) |. "
                "If the input looks messy, normalize key findings first.\n\n"
                + combined
            ),
        })
        analysis = self._response(temperature=0.35, max_tokens=1400)
        self.conversation.append({"role": "assistant", "content": analysis})
        return analysis

    def generate_reports(self) -> str:
        unauthorized_msg = self._ensure_authorized()
        if unauthorized_msg:
            return unauthorized_msg

        self.conversation.append({
            "role": "user",
            "content": (
                "Generate a professional penetration testing report based on our discussion so far. "
                "Include:\n"
                "- Executive summary (non-technical)\n"
                "- Checklist of phases completed (Recon, Scanning, Enumeration, Vulnerability Analysis, Reporting)\n"
                "- Vulnerabilities in a markdown table sorted by severity\n"
                "- Impact explanations\n"
                "- High-level, authorization-bound exploitation theory and paths (no illegal instructions)\n"
                "- Mitigation recommendations\n"
                "- Conclusion"
            ),
        })
        report = self._response(temperature=0.4, max_tokens=1600)
        self.conversation.append({"role": "assistant", "content": report})
        return report


assistant = PenTestAssistant()


# ----------------------------
# Gradio UI
# ----------------------------
def chatbot_interface(user_input, history):
    return assistant.chat(user_input or "")

def scan_analyzer(files=None, text_input: str = ""):
    return assistant.analyze_scans(files, text_input or "")

def generate_report():
    return assistant.generate_reports()

def set_mode(mode):
    assistant.set_mode(mode)
    return f"Mode set to {mode}"

def set_stage(stage):
    assistant.set_stage(stage)
    return f"Set stage to {stage}"

def set_auth(authorized):
    assistant.set_authorized(bool(authorized))
    return "Authorization confirmed." if authorized else "Authorization revoked."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PenTest Assistant")
    gr.Markdown("Professional, ethical guidance for authorized penetration testing.")

    with gr.Row():
        mode_btn = gr.Radio(["beginner", "professional"], label="Select Mode", value="professional")
        mode_output = gr.Textbox(label="Mode Status", interactive=False)
        mode_btn.change(set_mode, inputs=mode_btn, outputs=mode_output)

        stage_btn = gr.Dropdown(
            ["Recon", "Scanning", "Enumeration", "Vulnerability Analysis", "Reporting"],
            label="Checklist Stage", value=None, allow_custom_value=False
        )
        stage_output = gr.Textbox(label="Stage Status", interactive=False)
        stage_btn.change(set_stage, inputs=stage_btn, outputs=stage_output)

    authorized_chk = gr.Checkbox(label="I confirm I am authorized to test the target(s).", value=False)
    auth_status = gr.Textbox(label="Authorization Status", interactive=False)
    authorized_chk.change(set_auth, inputs=authorized_chk, outputs=auth_status)

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=chatbot_interface,
            title="Interactive Chat",
            description="Ask anything about penetration testing. High-level guidance only without authorization."
        )

    with gr.Tab("Scan Analyzer"):
        with gr.Row():
            file_input = gr.File(
                label="Upload scan/log files or a zip (txt, log, xml, zip)",
                file_types=[".txt", ".log", ".xml", ".zip"],
                file_count="multiple"
            )
        text_input = gr.Textbox(
            label="Paste scan output here",
            placeholder="Paste scan output here",
            lines=18
        )
        output = gr.Textbox(label="Scan Analysis", lines=20)
        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(fn=scan_analyzer, inputs=[file_input, text_input], outputs=output)

    with gr.Tab("Report Generator"):
        report_btn = gr.Button("Generate Report")
        report_output = gr.Textbox(label="Final Report", lines=25)
        report_btn.click(fn=generate_report, outputs=report_output)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(concurrency_count=10).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        debug=False,
        share=False,
    )
