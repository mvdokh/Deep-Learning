"""
ClassGUI — A quick classification labeling tool.

Select a folder of images, pick a task (the set of car classes),
then rapidly label each image via a searchable dropdown.
Classified images are copied/moved into class-named subfolders.

Usage
-----
    python ClassGUI/app.py
"""

import json
import os
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


# ══════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════

class ClassGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Car Image Classifier")
        self.root.geometry("900x700")
        self.root.minsize(750, 600)

        # State
        self.image_folder: Path | None = None
        self.output_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.current_idx: int = 0
        self.car_classes: list[str] = self._load_classes()
        self.history: list[tuple[Path, str]] = []  # (image_path, label) for undo

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
        # Re-import config to pick up a freshly written car_classes.json
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

        ttk.Button(top_frame, text="Select Image Folder", command=self._select_folder).pack(side=tk.LEFT, padx=(0, 5))
        self.folder_var = tk.StringVar(value="No folder selected")
        ttk.Label(top_frame, textvariable=self.folder_var, foreground="gray").pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(top_frame, text="Reload Classes", command=self._reload_classes).pack(side=tk.RIGHT, padx=(5, 0))

        # ── Output folder ──
        out_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        out_frame.pack(fill=tk.X)

        ttk.Button(out_frame, text="Output Folder", command=self._select_output).pack(side=tk.LEFT, padx=(0, 5))
        self.output_var = tk.StringVar(value="Same as input (subfolders)")
        ttk.Label(out_frame, textvariable=self.output_var, foreground="gray").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Move vs copy toggle
        self.move_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(out_frame, text="Move (instead of copy)", variable=self.move_var).pack(side=tk.RIGHT)

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
        ttk.Label(info_frame, textvariable=self.file_info_var, font=("Helvetica", 11)).pack(side=tk.LEFT)

        self.counter_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.counter_var, font=("Helvetica", 11, "bold")).pack(side=tk.RIGHT)

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

        ttk.Button(btn_row, text="<< Prev (Left)", command=self._prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Skip (S)", command=self._skip_image).pack(side=tk.LEFT, padx=(0, 5))

        self.classify_btn = ttk.Button(btn_row, text="Classify & Next (Enter)", command=self._classify_current)
        self.classify_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)

        ttk.Button(btn_row, text="Undo (Ctrl+Z)", command=self._undo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="Next >> (Right)", command=self._next_image).pack(side=tk.LEFT)

        # ── Status bar ──
        status_frame = ttk.Frame(self.root, padding=(10, 5))
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Select a folder to begin.")
        ttk.Label(status_frame, textvariable=self.status_var, foreground="gray").pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT)

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
        self.output_folder = None
        self.output_var.set("Same as input (subfolders)")
        self._scan_images()

    def _select_output(self):
        path = filedialog.askdirectory(title="Select output folder for classified images")
        if not path:
            return
        self.output_folder = Path(path)
        self.output_var.set(str(self.output_folder))

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

        # Load and display
        try:
            img = Image.open(img_path)
            img.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            cw = self.canvas.winfo_width() or PREVIEW_SIZE[0]
            ch = self.canvas.winfo_height() or PREVIEW_SIZE[1]
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo_ref, anchor=tk.CENTER)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                250, 200, text=f"Cannot load image:\n{e}",
                fill="red", font=("Helvetica", 12),
            )

    # ──────────────────────────────────────────
    # Classification actions
    # ──────────────────────────────────────────

    def _classify_current(self):
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return

        label = self.class_var.get()
        if not label:
            messagebox.showwarning("No class selected", "Please select a class from the dropdown.")
            return

        img_path = self.image_paths[self.current_idx]
        dest_base = self.output_folder or self.image_folder
        dest_dir = dest_base / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / img_path.name

        # Handle name collisions
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            if self.move_var.get():
                shutil.move(str(img_path), str(dest_path))
            else:
                shutil.copy2(str(img_path), str(dest_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to {'move' if self.move_var.get() else 'copy'} file:\n{e}")
            return

        self.history.append((img_path, label))
        action = "Moved" if self.move_var.get() else "Copied"
        self.status_var.set(f"{action} {img_path.name} -> {label}/")

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

        img_path, label = self.history.pop()
        dest_base = self.output_folder or self.image_folder
        dest_dir = dest_base / label

        # Find the file in the destination (might have a _N suffix)
        candidates = list(dest_dir.glob(f"{img_path.stem}*{img_path.suffix}"))
        if candidates:
            moved_file = candidates[-1]  # most recent
            if self.move_var.get():
                shutil.move(str(moved_file), str(img_path))
            else:
                moved_file.unlink()
            self.status_var.set(f"Undid: {img_path.name} from {label}/")
        else:
            self.status_var.set(f"Could not find file to undo for {img_path.name}")

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
