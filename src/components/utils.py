from pathlib import Path

def load_prompt(file_path: Path) -> tuple[str, str]:
    with open(file_path, mode="r") as f:
        txt = f.read()
        sys_msg, user_msg = txt.split("&&&\n")
        return sys_msg, user_msg