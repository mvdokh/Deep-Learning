"""
ClassGUI — A quick classification labeling tool.

Select a folder of images, pick a class from the dropdown, and classify.

- Raw images are saved to  data/raw/
- Processed images + a labels.json mapping are saved to  data/processed/

Usage
-----
    python ClassGUI/app.py
"""

import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# ──────────────────────────────────────────────
# Resolve project paths so we can import config
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
PREVIEW_SIZE = (500, 400)

RAW_DIR = config.RAW_DIR             # data/raw/
PROCESSED_DIR = config.PROCESSED_DIR  # data/processed/
LABELS_JSON = PROCESSED_DIR / "labels.json"


# ══════════════════════════════════════════════
# Labels JSON helpers
# ══════════════════════════════════════════════

def _load_labels() -> dict:
    """Load the existing labels.json or return an empty dict."""
    if LABELS_JSON.exists():
        return json.loads(LABELS_JSON.read_text())
    return {}


def _save_labels(labels: dict):
    """Write labels dict to labels.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_JSON.write_text(json.dumps(labels, indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════

class ClassGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Car Image Classifier")
        self.root.geometry("900x720")
        self.root.minsize(750, 620)

        # State
        self.image_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.current_idx: int = 0
        self.car_classes: list[str] = self._load_classes()
        self.labels: dict = _load_labels()
        self.history: list[tuple[str, str | None]] = []  # (filename, old_label_or_None)

        self._build_ui()
        self._bind_shortcuts()

    # ──────────────────────────────────────────
    # Class loading
    # ──────────────────────────────────────────

    def _load_classes(self) -> list[str]:
        """Load class labels from config (which reads car_classes.json if present)."""
        return [entry["label"] for entry in config.CAR_CLASSES]

    def _reload_classes(self):
        """Re-read classes (e.g. after running fetch_car_classes)."""
        import importlib
        importlib.reload(config)
        self.car_classes = self._load_classes()
        self._update_dropdown_values()
        self.status_var.set(f"Reloaded {len(self.car_classes)} classes")

    # ──────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar: folder selection ──
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="Select Image Folder",
                   command=self._select_folder).pack(side=tk.LEFT, padx=(0, 5))
        self.folder_var = tk.StringVar(value="No folder selected")
        ttk.Label(top_frame, textvariable=self.folder_var,
                  foreground="gray").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(top_frame, text="Reload Classes",
                   command=self._reload_classes).pack(side=tk.RIGHT, padx=(5, 0))

        # ── Output info ──
        out_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        out_frame.pack(fill=tk.X)

        ttk.Label(out_frame, text="Raw:", foreground="gray").pack(side=tk.LEFT)
        ttk.Label(out_frame, text=str(RAW_DIR),
                  foreground="steelblue").pack(side=tk.LEFT, padx=(3, 15))
        ttk.Label(out_frame, text="Processed:", foreground="gray").pack(side=tk.LEFT)
        ttk.Label(out_frame, text=str(PROCESSED_DIR),
                  foreground="steelblue").pack(side=tk.LEFT, padx=(3, 0))

        # ── Image preview ──
        preview_frame = ttk.LabelFrame(self.root, text="Image Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(preview_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._photo_ref = None  # prevent GC

        # ── File info ──
        info_frame = ttk.Frame(self.root, padding=(10, 0))
        info_frame.pack(fill=tk.X)

        self.file_info_var = tk.StringVar(value="No image loaded")
        ttk.Label(info_frame, textvariable=self.file_info_var,
                  font=("Helvetica", 11)).pack(side=tk.LEFT)

        self.counter_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.counter_var,
                  font=("Helvetica", 11, "bold")).pack(side=tk.RIGHT)

        # ── Classification controls ──
        class_frame = ttk.LabelFrame(self.root, text="Classify", padding=10)
        class_frame.pack(fill=tk.X, padx=10, pady=5)

        # Search / filter entry
        search_row = ttk.Frame(class_frame)
        search_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_row, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_changed)
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dropdown
        dropdown_row = ttk.Frame(class_frame)
        dropdown_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(dropdown_row, text="Class:").pack(side=tk.LEFT, padx=(0, 5))
        self.class_var = tk.StringVar()
        self.dropdown = ttk.Combobox(
            dropdown_row, textvariable=self.class_var,
            values=self.car_classes, state="readonly", width=50,
        )
        self.dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.car_classes:
            self.dropdown.current(0)

        # Buttons row
        btn_row = ttk.Frame(class_frame)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="<< Prev (Left)",
                   command=self._prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Skip (S)",
                   command=self._skip_image).pack(side=tk.LEFT, padx=(0, 5))

        self.classify_btn = ttk.Button(
            btn_row, text="Classify & Next (Enter)", command=self._classify_current)
        self.classify_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)

        ttk.Button(btn_row, text="Undo (Ctrl+Z)",
                   command=self._undo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Next >> (Right)",
                   command=self._next_image).pack(side=tk.LEFT)

        # ── Status bar ──
        status_frame = ttk.Frame(self.root, padding=(10, 5))
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Select a folder to begin.")
        ttk.Label(status_frame, textvariable=self.status_var,
                  foreground="gray").pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT)

        # Show label count
        n_labeled = len(self.labels)
        if n_labeled:
            self.status_var.set(
                f"Loaded {n_labeled} existing labels from labels.json. Select a folder to continue."
            )

    # ──────────────────────────────────────────
    # Keyboard shortcuts
    # ──────────────────────────────────────────

    def _bind_shortcuts(self):
        self.root.bind("<Return>", lambda e: self._classify_current())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<s>", lambda e: self._skip_image())
        self.root.bind("<S>", lambda e: self._skip_image())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Command-z>", lambda e: self._undo())  # macOS

    # ──────────────────────────────────────────
    # Folder selection
    # ──────────────────────────────────────────

    def _select_folder(self):
        path = filedialog.askdirectory(title="Select folder with car images")
        if not path:
            return
        self.image_folder = Path(path)
        self._scan_images()

    def _scan_images(self):
        """Scan the selected folder for image files."""
        self.image_paths = sorted(
            p for p in self.image_folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        self.current_idx = 0
        self.history.clear()
        self.folder_var.set(str(self.image_folder))

        if not self.image_paths:
            self.status_var.set("No images found in selected folder.")
            self.file_info_var.set("No image loaded")
            self.counter_var.set("")
            self.canvas.delete("all")
            return

        self.status_var.set(f"Found {len(self.image_paths)} images. Start classifying!")
        self._show_current_image()

    # ──────────────────────────────────────────
    # Image display
    # ──────────────────────────────────────────

    def _show_current_image(self):
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 250,
                self.canvas.winfo_height() // 2 or 200,
                text="All images processed!",
                fill="white", font=("Helvetica", 18),
            )
            self.file_info_var.set("Done")
            self.counter_var.set(f"{len(self.image_paths)}/{len(self.image_paths)}")
            self.progress["value"] = 100
            return

        img_path = self.image_paths[self.current_idx]

        # Update info
        self.file_info_var.set(img_path.name)
        self.counter_var.set(f"{self.current_idx + 1} / {len(self.image_paths)}")
        self.progress["value"] = (self.current_idx / max(len(self.image_paths), 1)) * 100

        # If this image already has a label, pre-select it in the dropdown
        existing_label = self.labels.get(img_path.name)
        if existing_label and existing_label in self.car_classes:
            self.class_var.set(existing_label)

        # Load and display
        try:
            img = Image.open(img_path)
            img.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            cw = self.canvas.winfo_width() or PREVIEW_SIZE[0]
            ch = self.canvas.winfo_height() or PREVIEW_SIZE[1]
            self.canvas.create_image(
                cw // 2, ch // 2, image=self._photo_ref, anchor=tk.CENTER)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                250, 200, text=f"Cannot load image:\n{e}",
                fill="red", font=("Helvetica", 12),
            )

    # ──────────────────────────────────────────
    # Classification actions
    # ──────────────────────────────────────────

    def _unique_dest(self, directory: Path, name: str) -> Path:
        """Return a non-colliding path inside `directory`."""
        dest = directory / name
        if not dest.exists():
            return dest
        stem = Path(name).stem
        suffix = Path(name).suffix
        counter = 1
        while dest.exists():
            dest = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest

    def _classify_current(self):
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return

        label = self.class_var.get()
        if not label:
            messagebox.showwarning(
                "No class selected", "Please select a class from the dropdown.")
            return

        img_path = self.image_paths[self.current_idx]
        filename = img_path.name

        # ── 1. Copy to data/raw/ (flat, preserving original) ──
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_dest = self._unique_dest(RAW_DIR, filename)
        try:
            shutil.copy2(str(img_path), str(raw_dest))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to raw/:\n{e}")
            return

        # Use the final filename (may have been de-duped)
        saved_name = raw_dest.name

        # ── 2. Copy to data/processed/ (flat) ──
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        proc_dest = self._unique_dest(PROCESSED_DIR, saved_name)
        try:
            shutil.copy2(str(img_path), str(proc_dest))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to processed/:\n{e}")
            return

        # ── 3. Update labels.json ──
        old_label = self.labels.get(saved_name)
        self.labels[proc_dest.name] = label
        _save_labels(self.labels)

        # ── 4. Record for undo ──
        self.history.append((saved_name, old_label))

        self.status_var.set(f"Classified {saved_name} -> {label}")
        self.current_idx += 1
        self._show_current_image()

    def _skip_image(self):
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return
        self.status_var.set(f"Skipped {self.image_paths[self.current_idx].name}")
        self.current_idx += 1
        self._show_current_image()

    def _next_image(self):
        if not self.image_paths:
            return
        self.current_idx = min(self.current_idx + 1, len(self.image_paths) - 1)
        self._show_current_image()

    def _prev_image(self):
        if not self.image_paths:
            return
        self.current_idx = max(self.current_idx - 1, 0)
        self._show_current_image()

    def _undo(self):
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return

        saved_name, old_label = self.history.pop()

        # Remove from raw/
        raw_file = RAW_DIR / saved_name
        if raw_file.exists():
            raw_file.unlink()

        # Remove from processed/
        proc_file = PROCESSED_DIR / saved_name
        if proc_file.exists():
            proc_file.unlink()

        # Restore label or remove entry
        if old_label is not None:
            self.labels[saved_name] = old_label
        else:
            self.labels.pop(saved_name, None)
        _save_labels(self.labels)

        self.status_var.set(f"Undid classification of {saved_name}")
        self.current_idx = max(self.current_idx - 1, 0)
        self._show_current_image()

    # ──────────────────────────────────────────
    # Dropdown search / filter
    # ──────────────────────────────────────────

    def _on_search_changed(self, *_args):
        query = self.search_var.get().lower().strip()
        if not query:
            self.dropdown["values"] = self.car_classes
        else:
            filtered = [c for c in self.car_classes if query in c.lower()]
            self.dropdown["values"] = filtered
            if filtered:
                self.dropdown.set(filtered[0])

    def _update_dropdown_values(self):
        self.dropdown["values"] = self.car_classes
        if self.car_classes:
            self.dropdown.current(0)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Try to set a modern theme
    style = ttk.Style(root)
    available_themes = style.theme_names()
    for theme in ("clam", "aqua", "vista", "alt"):
        if theme in available_themes:
            style.theme_use(theme)
            break

    app = ClassGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
