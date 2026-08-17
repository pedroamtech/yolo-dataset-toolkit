"""
main.py — YOLO Dataset Toolkit launcher

Runs any tool under tools/ independently, based on the task you pick.

Usage:
    python main.py                                   # interactive menu
    python main.py <tool_id> [args...]                # run a tool directly,
                                                        # passing args straight
                                                        # through to it
    python main.py --list                             # list tool ids

Examples:
    python main.py clean_dataset path/to/dataset --delete
    python main.py validate_labels path/to/dataset --num-classes 2
    python main.py analyze_size_distribution --save
"""

import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "tools"

# Each entry describes one task the toolkit can run and how to collect the
# arguments it needs. "script" must exist under tools/.
TOOLS = [
    {
        "id": "clean_dataset",
        "title": "Limpiar dataset — elimina imágenes sin anotaciones de persona (clase 0)",
        "script": "clean_dataset.py",
        "kind": "dataset_dir",
        "path_prompt": "Carpeta del dataset (contiene images/ y labels/)",
        "ask_delete": True,
    },
    {
        "id": "validate_labels",
        "title": "Validar etiquetas — detecta clases fuera de rango, filas mal formadas, "
                 "coordenadas NaN/fuera de límites y archivos huérfanos",
        "script": "validate_labels.py",
        "kind": "dataset_dir",
        "path_prompt": "Carpeta raíz del dataset (se buscan subcarpetas 'labels/')",
        "ask_num_classes": True,
    },
    {
        "id": "normalize_labels",
        "title": "Normalizar etiquetas — recorta coordenadas YOLO a [0,1] y descarta cajas de área cero",
        "script": "normalize_manipal_labels.py",
        "kind": "dataset_dir",
        "path_prompt": "Carpeta raíz del dataset (con subcarpetas train/val/test)",
    },
    {
        "id": "yolo_person_labeler",
        "title": "Etiquetar/editar personas — detección automática (HOG) + edición manual "
                 "con zoom/pan y colores por clase",
        "script": "yolo_person_labeler.py",
        "kind": "dataset_dir",
        "path_prompt": "Carpeta del dataset (contiene images/ y labels/)",
    },
    {
        "id": "rename_images",
        "title": "Renombrar imágenes — antepone un prefijo fijo a todas las imágenes de una carpeta",
        "script": "rename_images.py",
        "kind": "images_dir",
        "path_prompt": "Carpeta de imágenes",
        "ask_prefix": True,
    },
    {
        "id": "video_to_frames",
        "title": "Extraer frames — convierte cada vídeo de una carpeta en una subcarpeta de frames",
        "script": "video_to_frames.py",
        "kind": "images_dir",
        "path_prompt": "Carpeta que contiene los vídeos",
    },
    {
        "id": "analyze_size_distribution",
        "title": "Analizar tamaños — distribución de tamaño de objetos (AS/RS, ajuste log-normal, CCDF)",
        "script": "analyze_size_distribution.py",
        "kind": "labels_images",
    },
]


def find_tool(token: str):
    token = token.strip().lower()
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(TOOLS):
            return TOOLS[idx]
        return None
    for tool in TOOLS:
        if tool["id"] == token:
            return tool
    return None


def print_menu():
    print("\nYOLO Dataset Toolkit — selecciona una tarea:\n")
    for i, tool in enumerate(TOOLS, start=1):
        print(f"  {i}. [{tool['id']}]  {tool['title']}")
    print("  0. Salir\n")


def run_tool(tool: dict, extra_args: list) -> int:
    script_path = TOOLS_DIR / tool["script"]
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def prompt_path(prompt: str, allow_empty: bool = True) -> str:
    suffix = " (Enter para diálogo de selección)" if allow_empty else ""
    value = input(f"{prompt}{suffix}: ").strip().strip('"')
    return value


def interactive_run(tool: dict) -> int:
    args = []

    if tool["kind"] in ("dataset_dir", "images_dir"):
        path = prompt_path(tool["path_prompt"])
        if path:
            args.append(path)

        if tool.get("ask_delete"):
            ans = input("¿Eliminar permanentemente en vez de mover a _removed/? [y/N]: ").strip().lower()
            if ans in ("y", "yes", "s", "si", "sí"):
                args.append("--delete")

        if tool.get("ask_num_classes"):
            if not path:
                print("[ERROR] validate_labels necesita una ruta de dataset (no soporta diálogo).")
                return 1
            while True:
                n = input("Número de clases del dataset: ").strip()
                if n.isdigit():
                    args += ["--num-classes", n]
                    break
                print("  Ingresa un entero válido.")

        if tool.get("ask_prefix"):
            prefix = input("Prefijo a anteponer (Enter para ninguno): ").strip()
            if prefix:
                args += ["--prefix", prefix]

    elif tool["kind"] == "labels_images":
        labels = prompt_path("Carpeta de labels YOLO (.txt)")
        if labels:
            images = prompt_path("Carpeta de imágenes correspondiente", allow_empty=False)
            if not images:
                print("[ERROR] analyze_size_distribution necesita --images junto con --labels.")
                return 1
            args += ["--labels", labels, "--images", images]
        ans = input("¿Guardar la figura como PNG? [y/N]: ").strip().lower()
        if ans in ("y", "yes", "s", "si", "sí"):
            args.append("--save")

    return run_tool(tool, args)


def main():
    argv = sys.argv[1:]

    if not argv:
        while True:
            print_menu()
            choice = input("Selección: ").strip()
            if choice in ("0", "q", "quit", "exit"):
                return
            tool = find_tool(choice)
            if not tool:
                print(f"Opción inválida: '{choice}'")
                continue
            interactive_run(tool)
        return

    if argv[0] in ("-h", "--help"):
        print(__doc__)
        return

    if argv[0] == "--list":
        for tool in TOOLS:
            print(f"{tool['id']:<28} {tool['title']}")
        return

    tool = find_tool(argv[0])
    if not tool:
        print(f"Herramienta desconocida: '{argv[0]}'")
        print("Usa 'python main.py --list' para ver las opciones disponibles.")
        raise SystemExit(1)

    raise SystemExit(run_tool(tool, argv[1:]))


if __name__ == "__main__":
    main()
