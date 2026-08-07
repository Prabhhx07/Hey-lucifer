import sys
import json
import time
import argparse

import subprocess

from ax_agent import (
    get_pid_by_app_name,
    get_frontmost_pid,
    get_main_window,
    walk,
    activate_app,
    click_element_directly,
    type_into_element_reliable,
    AXUIElementCreateApplication,
    INTERESTING_ROLES,
)

from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"  
TYPABLE_ROLES = {"AXTextArea", "AXTextField", "AXComboBox"}
MAX_STEPS = 15  

_model = None
_tokenizer = None


def get_model():
    """Lazy-load the model once per process (loading takes a few seconds)."""
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {MODEL_NAME} (first run downloads weights, ~2GB)...")
        _model, _tokenizer = load(MODEL_NAME)
    return _model, _tokenizer


def ensure_window_open(pid, app_name):
    app_element = AXUIElementCreateApplication(pid)
    window = get_main_window(app_element)
    if window is not None:
        return True

    print(f"{app_name} has no open window. Attempting Cmd+N to open one...")
    activate_app(pid)
    time.sleep(0.4)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "n" using command down'
    ])
    time.sleep(1.0)

    app_element = AXUIElementCreateApplication(pid)
    window = get_main_window(app_element)
    if window is None:
        print(f"Still no window after Cmd+N — {app_name} may need manual attention.")
        return False

    print("Window opened successfully.")
    return True


def get_current_elements(pid, max_elements=60, retries=3, retry_delay=0.4):
    
    app_element = AXUIElementCreateApplication(pid)

    window = None
    for attempt in range(retries):
        window = get_main_window(app_element)
        if window is not None:
            break
        time.sleep(retry_delay)

    if window is None:
        print("Warning: no window found after retries — falling back to "
              "app root, which mostly exposes menu bar items. Perception "
              "will likely be too sparse to act on.")

    root = window if window is not None else app_element

    elements = []
    for path, desc, element in walk(root):
        role = desc.get("AXRole")
        if role not in INTERESTING_ROLES:
            continue
        title = desc.get("AXTitle") or desc.get("AXValue") or desc.get("AXDescription") or ""
        elements.append({
            "index": len(elements),
            "role": role,
            "title": title,
            "path": path,
            "element": element,
        })
        if len(elements) >= max_elements:
            break
    return elements


def format_elements_for_prompt(elements):
    lines = []
    for e in elements:
        title = e["title"].strip()
        title_part = f': "{title}"' if title else ""
        typable_hint = " (editable — can type here)" if e["role"] in TYPABLE_ROLES else ""
        lines.append(f'  [{e["index"]}] {e["role"]}{typable_hint}{title_part}')
    return "\n".join(lines) if lines else "  (no interactive elements found)"


SYSTEM_PROMPT = """You are a computer-control agent. You are given a goal, a \
history of actions already taken, and a numbered list of the currently \
visible interactive UI elements. Decide the single next action to make \
progress toward the goal.

Respond with ONLY a JSON object, no other text, in one of these forms:

To click an element:
{"action": "click", "index": <element index>, "reason": "<why>"}

To type text into an element (this will click it first, then type):
{"action": "type", "index": <element index>, "text": "<text to type>", "reason": "<why>"}

To declare the task complete:
{"action": "done", "reason": "<why the goal is achieved>"}

To declare the task stuck (no useful element to progress with):
{"action": "stuck", "reason": "<what's missing>"}

Rules:
- Only reference indices that appear in the current element list.
- Prefer the most direct action toward the goal.
- If you already typed the needed text and it's visible, use "done".
"""


def build_user_prompt(goal, elements, history, blocked_indices=None):
    element_list = format_elements_for_prompt(elements)
    history_text = "\n".join(
        f'  Step {i+1}: {h}' for i, h in enumerate(history)
    ) if history else "  (no actions taken yet)"

    editable = [e for e in elements if e["role"] in TYPABLE_ROLES]
    if editable:
        editable_lines = "\n".join(
            f'  [{e["index"]}] {e["role"]}' + (f': "{e["title"].strip()}"' if e["title"].strip() else " (empty)")
            for e in editable
        )
        editable_block = f"\nElements you CAN type into (choose ONLY from these for \"type\" actions):\n{editable_lines}\n"
    else:
        editable_block = "\nNo editable elements are currently visible — do not use \"type\".\n"

    blocked_block = ""
    if blocked_indices:
        blocked_block = (f"\nDo NOT choose these indices again — they were already "
                          f"tried and rejected: {sorted(blocked_indices)}\n")

    return f"""Goal: {goal}

Actions taken so far:
{history_text}
{blocked_block}
Currently visible elements:
{element_list}
{editable_block}
What is the next action?"""


def call_model(goal, elements, history, blocked_indices=None):
    model, tokenizer = get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(goal, elements, history, blocked_indices)},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
    return response


def parse_action(response_text):
    """Extract the JSON action object from the model's response, tolerating
    minor formatting noise (e.g. accidental markdown fences)."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {response_text!r}")

    return json.loads(text[start:end + 1])


def execute_action(action, elements, pid):
    """Execute a parsed action against the current element list. Returns
    a short human-readable description for the history log."""
    kind = action.get("action")

    if kind == "click":
        idx = action.get("index")
        target = next((e for e in elements if e["index"] == idx), None)
        if target is None:
            return f"FAILED: index {idx} not found in current elements", False
        activate_app(pid)
        time.sleep(0.3)
        ok = click_element_directly(target["element"])
        return f'Clicked [{idx}] {target["role"]}: "{target["title"]}" (success={ok})', False

    elif kind == "type":
        idx = action.get("index")
        text = action.get("text", "")
        target = next((e for e in elements if e["index"] == idx), None)
        if target is None:
            return f"FAILED: index {idx} not found in current elements", False

        if target["role"] not in TYPABLE_ROLES:
            return (f'REJECTED: [{idx}] is {target["role"]}, which cannot accept '
                     f'text input. Choose an element marked "(editable)" instead — '
                     f'typically AXTextArea or AXTextField.'), False

        ok = type_into_element_reliable(target["element"], text, pid=pid)
        return f'Typed "{text}" into [{idx}] {target["role"]} (verified={ok})', False

    elif kind == "done":
        return f'DONE: {action.get("reason", "")}', True

    elif kind == "stuck":
        return f'STUCK: {action.get("reason", "")}', True

    else:
        return f"FAILED: unrecognized action {action!r}", False


def run_task(app_name, goal, max_steps=MAX_STEPS):
    if app_name:
        pid, name = get_pid_by_app_name(app_name)
        if pid is None:
            print(f"Could not find a running app matching '{app_name}'")
            sys.exit(1)
    else:
        pid, name = get_frontmost_pid()

    print(f"\n=== Starting task on {name} (pid {pid}) ===")
    print(f"Goal: {goal}\n")

    if not ensure_window_open(pid, name):
        print("Aborting: could not establish an open window to work with.")
        return

    history = []
    last_action_signature = None
    repeat_count = 0
    blocked_indices = set()  # indices confirmed non-typable — never retry these

    for step in range(1, max_steps + 1):
        print(f"--- Step {step}/{max_steps} ---")
        elements = get_current_elements(pid)

        try:
            raw_response = call_model(goal, elements, history, blocked_indices)
            action = parse_action(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Model response could not be parsed: {e}")
            history.append("FAILED: could not parse model response")
            continue

        print(f"Model chose: {action}")

        chosen_idx = action.get("index")
        if action.get("action") == "type" and chosen_idx in blocked_indices:
            result = (f'SKIPPED: [{chosen_idx}] was already confirmed non-typable — '
                      f'not retrying. Pick a different index.')
            print(result)
            history.append(result)
            continue

        signature = (action.get("action"), action.get("index"))
        if signature == last_action_signature:
            repeat_count += 1
        else:
            repeat_count = 0
        last_action_signature = signature

        if repeat_count >= 2:  
            print(f"\n=== Stopped: repeated the same action {repeat_count + 1}x "
                  f"without adapting — likely stuck ===")
            return

        result, is_terminal = execute_action(action, elements, pid)
        print(result)
        history.append(result)

        if result.startswith("REJECTED") and chosen_idx is not None:
            blocked_indices.add(chosen_idx)

        if is_terminal:
            print(f"\n=== Task ended after {step} step(s) ===")
            return

        time.sleep(0.3)

    print(f"\n=== Hit max steps ({max_steps}) without completion ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", help="App name to control (default: frontmost app)")
    parser.add_argument("--goal", required=True, help="Natural-language goal for the agent")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    run_task(args.app, args.goal, max_steps=args.max_steps)


if __name__ == "__main__":
    main()