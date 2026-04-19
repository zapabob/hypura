# EasyNovelAssistant Backend Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let EasyNovelAssistant switch between KoboldCpp and Hypura while keeping the existing Kobold-compatible generation flow, partial-output polling, and abort behavior.

**Architecture:** Keep one EasyNovelAssistant runtime object at `ctx.kobold_cpp`, but convert it into a backend-selection facade. The facade delegates backend-specific launch logic to KoboldCpp and Hypura command builders while both backends continue using the same Kobold-compatible HTTP routes for model lookup, generation, polling, and abort.

**Tech Stack:** Python, Tkinter, requests, Windows batch launchers, Hypura KoboldCpp compatibility CLI

---

### Task 1: Add backend-selection configuration and facade behavior

**Files:**
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\setup\res\default_config.json`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\kobold_cpp.py`
- Test: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py`

- [ ] **Step 1: Write the failing test**

```python
from kobold_cpp import BackendSelectionMixin


def test_backend_selection_defaults_to_koboldcpp():
    ctx = {"llm_backend": None}
    assert BackendSelectionMixin.normalize_backend_name(ctx) == "koboldcpp"


def test_backend_selection_accepts_hypura():
    ctx = {"llm_backend": "Hypura"}
    assert BackendSelectionMixin.normalize_backend_name(ctx) == "hypura"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py -v`

Expected: FAIL because `BackendSelectionMixin` and backend normalization helpers do not exist yet.

- [ ] **Step 3: Add backend-selection implementation**

```python
class BackendSelectionMixin:
    @staticmethod
    def normalize_backend_name(ctx):
        raw = ctx.get("llm_backend")
        if raw is None:
            return "koboldcpp"
        value = str(raw).strip().lower()
        return "hypura" if value == "hypura" else "koboldcpp"
```

```python
class KoboldCpp:
    def __init__(self, ctx):
        self.ctx = ctx
        self.backend = self.normalize_backend_name(ctx)
        self.base_url = f"http://{ctx['koboldcpp_host']}:{ctx['koboldcpp_port']}"
        self.model_url = f"{self.base_url}/api/v1/model"
        self.generate_url = f"{self.base_url}/api/v1/generate"
        self.stream_url = f"{self.base_url}/api/extra/generate/stream"
        self.check_url = f"{self.base_url}/api/extra/generate/check"
        self.abort_url = f"{self.base_url}/api/extra/abort"
```

- [ ] **Step 4: Add `hypura_path` default**

```json
{
  "llm_backend": "koboldcpp",
  "hypura_path": ""
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add EasyNovelAssistant/setup/res/default_config.json EasyNovelAssistant/src/kobold_cpp.py EasyNovelAssistant/tests/test_backend_selection.py
git commit -m "feat(easy-novel-assistant): add backend selection config"
```

### Task 2: Implement backend-specific launch commands and preserve the shared HTTP flow

**Files:**
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\kobold_cpp.py`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\menu\model_menu.py`
- Test: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_launch.py`

- [ ] **Step 1: Write the failing test**

```python
from kobold_cpp import build_hypura_command


def test_build_hypura_command_uses_compat_mode():
    command = build_hypura_command(
        executable="hypura",
        model_path="C:/models/demo.gguf",
        host="127.0.0.1",
        port=5001,
        context_size=8192,
    )
    assert command[:3] == ["hypura", "koboldcpp", "C:/models/demo.gguf"]
    assert "--host" in command
    assert "--port" in command
    assert "--context" in command
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_launch.py -v`

Expected: FAIL because the Hypura command builder does not exist yet.

- [ ] **Step 3: Implement launch builders and backend-specific launch flow**

```python
def build_hypura_command(executable, model_path, host, port, context_size):
    return [
        executable,
        "koboldcpp",
        model_path,
        "--host",
        str(host),
        "--port",
        str(port),
        "--context",
        str(context_size),
    ]
```

```python
def launch_server(self):
    loaded_model = self.get_model()
    if loaded_model is not None:
        return f"{loaded_model} がすでにロード済みです。"

    llm_path = self.resolve_model_path(...)
    if self.backend == "hypura":
        command = build_hypura_command(...)
        subprocess.Popen(command, cwd=os.getcwd())
        return self.wait_for_model_ready("Hypura")

    command = self.build_koboldcpp_command(...)
    subprocess.run(command, cwd=Path.kobold_cpp, shell=True)
    return None
```

- [ ] **Step 4: Ensure direct GGUF selection works for both backends**

```python
model_config = {
    "max_gpu_layer": gpu_layers,
    "context_size": 4096,
    "urls": [f"file://{target_path}"],
    "file_name": file_name,
    "local_file": True,
    "temporary": True,
}
```

```python
if self.ctx.kobold_cpp.backend == "hypura":
    target_path = file_path
```

- [ ] **Step 5: Run tests to verify launch-command logic passes**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_launch.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add EasyNovelAssistant/src/kobold_cpp.py EasyNovelAssistant/src/menu/model_menu.py EasyNovelAssistant/tests/test_backend_launch.py
git commit -m "feat(easy-novel-assistant): launch Hypura compat backend"
```

### Task 3: Add settings UI, title updates, launchers, and documentation

**Files:**
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\menu\setting_menu.py`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\form.py`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-KoboldCpp.bat`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-HypuraProxy.bat`
- Modify: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\README.md`

- [ ] **Step 1: Write the failing test**

```python
from form import format_window_title


def test_window_title_includes_backend_label():
    title = format_window_title(
        base_title="EasyNovelAssistant",
        model_name="demo.gguf",
        backend_label="Hypura",
        generating=False,
        file_path=None,
    )
    assert "[Hypura]" in title
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py -v`

Expected: FAIL because the title-formatting helper does not exist yet.

- [ ] **Step 3: Implement settings menu and title updates**

```python
backend_menu = tk.Menu(self.menu, tearoff=False)
self.menu.add_cascade(label=f"LLMバックエンド: {self._backend_label()}", menu=backend_menu)
for backend_name in ("koboldcpp", "hypura"):
    backend_menu.add_radiobutton(
        label=self._display_backend_name(backend_name),
        value=backend_name,
        variable=backend_var,
        command=lambda b=backend_name: self._set_backend(b),
    )
```

```python
def update_title(self):
    backend_label = self.ctx.kobold_cpp.display_backend_name()
    title = f"EasyNovelAssistant [{backend_label}]"
    if self.ctx.kobold_cpp.model_name is not None:
        title += f" - {self.ctx.kobold_cpp.model_name}"
```

- [ ] **Step 4: Update launchers and docs**

```bat
echo [INFO] Launching EasyNovelAssistant with llm_backend=hypura
```

```markdown
- `設定 -> LLMバックエンド` から `KoboldCpp` または `Hypura` を選択できます。
- `Hypura` 選択時は `hypura koboldcpp <model>` を使って互換モードで起動します。
```

- [ ] **Step 5: Run tests to verify it passes**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add EasyNovelAssistant/src/menu/setting_menu.py EasyNovelAssistant/src/form.py Run-EasyNovelAssistant-KoboldCpp.bat Run-EasyNovelAssistant-HypuraProxy.bat README.md
git commit -m "feat(easy-novel-assistant): expose backend selector in UI"
```

### Task 4: Verify runtime behavior and record implementation evidence

**Files:**
- Create: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\_docs\2026-04-19_easy-novel-assistant_backend-selection_codex.md`

- [ ] **Step 1: Run Python tests**

Run: `py -3 -m pytest C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_launch.py -v`

Expected: PASS

- [ ] **Step 2: Smoke-test KoboldCpp mode**

Run: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-KoboldCpp.bat`

Expected: Existing KoboldCpp endpoint check succeeds and EasyNovelAssistant launches.

- [ ] **Step 3: Smoke-test Hypura mode**

Run: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-HypuraProxy.bat`

Expected: EasyNovelAssistant launches with `llm_backend=hypura` and the compat endpoint becomes reachable.

- [ ] **Step 4: Write implementation log**

```markdown
# Overview
- Added backend selection between KoboldCpp and Hypura.

# Changed files
- EasyNovelAssistant/src/kobold_cpp.py
- EasyNovelAssistant/src/menu/setting_menu.py
- EasyNovelAssistant/src/form.py

# Verification
- pytest ...
- launcher smoke checks ...
```

- [ ] **Step 5: Commit**

```bash
git add _docs/2026-04-19_easy-novel-assistant_backend-selection_codex.md
git commit -m "docs: add backend selection implementation log"
```
