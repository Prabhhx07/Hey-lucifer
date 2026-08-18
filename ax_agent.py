"""
Milestone 1 — Accessibility API Perception & Control
------------------------------------------------------
Reads structured UI element data from a macOS app, and provides functions
to click/focus/type into elements.

Run this ON YOUR MAC (not in a sandbox) — it needs pyobjc and macOS
Accessibility permissions.

Setup:
    pip install pyobjc-framework-ApplicationServices pyobjc-framework-Quartz

Then grant Accessibility permission:
    System Settings -> Privacy & Security -> Accessibility
    -> add Terminal (or your IDE / Python interpreter) and enable it.

Usage:
    python ax_agent.py                 # dumps UI tree of frontmost app
    python ax_agent.py --app "Notes"   # dumps UI tree of a named app
"""

import sys
import time
import argparse
import subprocess

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyAttributeNames,
    AXUIElementPerformAction,
    AXUIElementSetAttributeValue,
    kAXErrorSuccess,
)
from AppKit import (
    NSWorkspace,
    NSRunningApplication,
    NSApplicationActivateIgnoringOtherApps,
)
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGHIDEventTap,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
    kCGEventFlagMaskCommand,
)

# Attributes we care about when describing an element
INTERESTING_ATTRS = [
    "AXRole",
    "AXTitle",
    "AXValue",
    "AXDescription",
    "AXPosition",
    "AXSize",
    "AXEnabled",
]

MAX_DEPTH = 8  # avoid runaway recursion into huge UI trees

# Roles worth keeping when filtering — interactive/content elements,
# not decorative wrappers or menu bar chrome.
INTERESTING_ROLES = {
    "AXButton",
    "AXTextField",
    "AXTextArea",
    "AXStaticText",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXComboBox",
    "AXLink",
    "AXMenuButton",
    "AXTable",
    "AXRow",
    "AXCell",
    "AXList",
    "AXOutline",
    "AXImage",
    "AXTabGroup",
    "AXToolbar",
    "AXSlider",
}

# Virtual keycodes (US layout) used for the select-all + delete sequence.
_VK_A = 0
_VK_DELETE = 51


def get_frontmost_pid():
    """Return the PID of the frontmost application."""
    ws = NSWorkspace.sharedWorkspace()
    app = ws.frontmostApplication()
    return app.processIdentifier(), app.localizedName()


def get_pid_by_app_name(name):
    """Find a running app's PID by (partial, case-insensitive) name."""
    ws = NSWorkspace.sharedWorkspace()
    for app in ws.runningApplications():
        app_name = app.localizedName() or ""
        if name.lower() in app_name.lower():
            return app.processIdentifier(), app_name
    return None, None


def get_attr(element, attr):
    """Safely read an accessibility attribute, returning None on failure."""
    err, value = AXUIElementCopyAttributeValue(element, attr, None)
    if err != kAXErrorSuccess:
        return None
    return value


def describe_element(element):
    """Return a dict of the interesting attributes for one element."""
    desc = {}
    for attr in INTERESTING_ATTRS:
        val = get_attr(element, attr)
        if val is not None:
            desc[attr] = str(val)
    return desc


def walk(element, depth=0, max_depth=MAX_DEPTH, path=""):
    """
    Recursively walk the accessibility tree, yielding (path, description, element)
    for each node. `path` is a simple index-based path -- treat it as valid only
    for the walk that produced it; always re-walk fresh before acting, since
    paths go stale the moment the UI state changes.
    """
    if depth > max_depth:
        return

    desc = describe_element(element)
    yield path, desc, element

    children = get_attr(element, "AXChildren") or []
    for i, child in enumerate(children):
        child_path = f"{path}.{i}" if path else str(i)
        yield from walk(child, depth + 1, max_depth, child_path)


def get_main_window(app_element):
    """
    Return the app's focused/main window element, or fall back to the
    first window in AXWindows if no window is marked focused/main.
    This skips the AXMenuBar branch entirely, which is usually the
    largest and least useful part of the tree for actual task execution.
    """
    focused = get_attr(app_element, "AXFocusedWindow")
    if focused is not None:
        return focused

    windows = get_attr(app_element, "AXWindows") or []
    for win in windows:
        is_main = get_attr(win, "AXMain")
        if is_main:
            return win

    return windows[0] if windows else None


def ensure_window_open(pid, app_name):
    """
    Check whether the app currently has a window before starting a task.
    Some apps (Notes included) can genuinely report zero windows while
    still running -- if so, attempt Cmd+N to open one. Without this
    upfront check, perception can start against an empty/near-empty
    element list with nothing meaningful to act on.
    """
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
        print(f"Still no window after Cmd+N -- {app_name} may need manual attention.")
        return False

    print("Window opened successfully.")
    return True


def dump_ui_tree(pid, app_name, filter_roles=True):
    app_element = AXUIElementCreateApplication(pid)
    window = get_main_window(app_element)

    if window is None:
        print(f"\nNo window found for {app_name} (pid {pid}) -- "
              f"is the app actually showing a window? Falling back to full app tree.\n")
        root = app_element
    else:
        root = window
        title = get_attr(window, "AXTitle") or "(untitled)"
        print(f"\n=== UI tree for {app_name} window: '{title}' (pid {pid}) ===\n")

    count = 0
    shown = 0
    for path, desc, _ in walk(root):
        role = desc.get("AXRole", "?")
        title = desc.get("AXTitle") or desc.get("AXValue") or desc.get("AXDescription") or ""

        if filter_roles and role not in INTERESTING_ROLES:
            count += 1
            continue

        if role or title:
            print(f"[{path}] {role}: {title}")
            shown += 1
        count += 1

    print(f"\n{shown} interesting elements shown, {count} total walked "
          f"(max_depth={MAX_DEPTH}, filter_roles={filter_roles}).")


def find_element_by_path(pid, path):
    """Re-locate an element using the path produced by walk(). Only valid
    against the same tree state the path came from -- prefer find_first_by_role
    for anything after the UI may have changed."""
    app_element = AXUIElementCreateApplication(pid)
    if path == "":
        return app_element
    for path_i, desc, element in walk(app_element):
        if path_i == path:
            return element
    return None


def find_first_by_role(pid, role, window=None):
    """
    Walk the app's current tree fresh and return the first element matching
    `role` (e.g. 'AXTextArea'). Always re-walk immediately before acting --
    cached paths from an earlier walk go stale as soon as the UI state
    changes (e.g. a different list item gets selected), since paths are
    positional indices, not stable IDs.
    """
    app_element = AXUIElementCreateApplication(pid)
    root = window if window is not None else (get_main_window(app_element) or app_element)
    for path, desc, element in walk(root):
        if desc.get("AXRole") == role:
            return path, element
    return None, None


def click_element(element):
    """Perform the default 'press' action on an element (button, menu item, etc.)."""
    err = AXUIElementPerformAction(element, "AXPress")
    return err == kAXErrorSuccess


def type_into_element(element, text):
    """Set the AXValue of a text field / text area directly (fast, but
    many Cocoa/SwiftUI text views silently ignore this -- always verify
    the result by re-reading AXValue, or prefer type_into_element_reliable()."""
    err = AXUIElementSetAttributeValue(element, "AXValue", text)
    return err == kAXErrorSuccess


def focus_element(element):
    """Set AXFocused on an element so subsequent keystrokes land in it."""
    err = AXUIElementSetAttributeValue(element, "AXFocused", True)
    return err == kAXErrorSuccess


def activate_app(pid):
    """
    Bring the target app to the actual OS foreground. This is separate
    from -- and required in addition to -- focus_element()'s AXFocused
    attribute: AXFocused only sets accessibility focus *within* the app,
    but CGEventPost keystrokes go to whatever app is frontmost at the OS
    level. Skipping this step is how keystrokes end up typed into the
    wrong window (e.g. your terminal) even when AXFocused succeeded.
    """
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if app is None:
        print(f"Warning: could not find running app for pid {pid}")
        return False
    return app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


def click_at_point(x, y):
    """Simulate a real left mouse click at absolute screen coordinates.
    More reliable than AXFocused for establishing genuine input focus --
    some apps (Notes among them) accept real clicks but silently ignore
    the AXFocused accessibility attribute."""
    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(0.05)
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, up)


def get_element_center(element):
    """Return the (x, y) screen center of an element, or None if position/size
    attributes aren't available."""
    pos = get_attr(element, "AXPosition")
    size = get_attr(element, "AXSize")
    if pos is None or size is None:
        return None
    # AXPosition/AXSize come back as AXValue-wrapped CGPoint/CGSize structs;
    # pyobjc exposes them with .x/.y and .width/.height.
    try:
        x = pos.x + size.width / 2
        y = pos.y + size.height / 2
        return (x, y)
    except AttributeError:
        return None


def click_element_directly(element):
    """Click the actual center of an element on screen. Use this instead
    of focus_element() for apps that ignore programmatic AXFocused writes."""
    center = get_element_center(element)
    if center is None:
        print("Warning: could not determine element position -- cannot click.")
        return False
    click_at_point(*center)
    return True


def type_via_keystrokes(text, delay=0.005):
    """
    Simulate real keyboard input by posting Unicode keyboard events at
    the OS level. Works regardless of whether the target text view
    honors direct AXValue writes, since from the app's point of view
    this looks identical to the user typing.

    IMPORTANT: the target element must already be focused/clicked before
    calling this -- these events go to whatever currently has keyboard
    focus, not to a specific element.
    """
    for char in text:
        event_down = CGEventCreateKeyboardEvent(None, 0, True)
        CGEventKeyboardSetUnicodeString(event_down, len(char), char)
        CGEventPost(kCGHIDEventTap, event_down)

        event_up = CGEventCreateKeyboardEvent(None, 0, False)
        CGEventKeyboardSetUnicodeString(event_up, len(char), char)
        CGEventPost(kCGHIDEventTap, event_up)

        time.sleep(delay)


def select_all_and_delete():
    """
    Select all existing text in whatever field currently has keyboard
    focus (Cmd+A) and delete it, so a subsequent type_via_keystrokes()
    call REPLACES the field's contents instead of appending to them.

    This is what was missing before: type_into_element_reliable() used
    to type directly at the current cursor position, so calling it twice
    (e.g. two separate phone commands against the same open note) would
    concatenate text instead of overwriting it -- "Hi" + "hello" becoming
    "Hihello" rather than "hello".
    """
    down = CGEventCreateKeyboardEvent(None, _VK_A, True)
    CGEventSetFlags(down, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, down)

    up = CGEventCreateKeyboardEvent(None, _VK_A, False)
    CGEventSetFlags(up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, up)

    time.sleep(0.05)

    down = CGEventCreateKeyboardEvent(None, _VK_DELETE, True)
    CGEventPost(kCGHIDEventTap, down)
    up = CGEventCreateKeyboardEvent(None, _VK_DELETE, False)
    CGEventPost(kCGHIDEventTap, up)

    time.sleep(0.05)


def type_into_element_reliable(element, text, verify=True, pid=None, replace=True):
    """
    Best-effort text entry: bring the app to the OS foreground, click the
    element to establish real focus, optionally clear any existing content
    (replace=True, the default -- set False if you deliberately want to
    append), then type via simulated keystrokes. Re-reads AXValue afterward
    to confirm the text actually landed, since AX return codes alone aren't
    trustworthy.

    `pid` is required to correctly activate the target app -- without it,
    keystrokes may go to whatever window currently has real OS focus
    (e.g. your terminal) instead of the intended app.
    """
    if pid is not None:
        activate_app(pid)
        time.sleep(0.4)  # give macOS time to actually switch foreground app
    else:
        print("Warning: no pid provided -- cannot guarantee the target app "
              "is frontmost. Keystrokes may go to the wrong window.")

    # Prefer a real click over AXFocused -- some apps (e.g. Notes) accept
    # clicks but silently ignore the accessibility focus attribute.
    if not click_element_directly(element):
        focus_element(element)  # fall back, better than nothing

    time.sleep(0.2)

    if replace:
        select_all_and_delete()

    type_via_keystrokes(text)

    if verify:
        time.sleep(0.2)
        new_value = get_attr(element, "AXValue") or ""
        if replace:
            matches = str(new_value) == text
        else:
            matches = text in str(new_value)
        if matches:
            return True
        else:
            print(f"Warning: verification failed. AXValue now reads: {new_value!r}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", help="App name to inspect (default: frontmost app)")
    parser.add_argument("--no-filter", action="store_true",
                         help="Show all elements, not just INTERESTING_ROLES")
    args = parser.parse_args()

    if args.app:
        pid, name = get_pid_by_app_name(args.app)
        if pid is None:
            print(f"Could not find a running app matching '{args.app}'")
            sys.exit(1)
    else:
        pid, name = get_frontmost_pid()

    dump_ui_tree(pid, name, filter_roles=not args.no_filter)


if __name__ == "__main__":
    main()