import os

structure = {
    "api": {
        "__init__.py": "",
        "main.py": "",
        "startup.py": "",
        "core": {
            "__init__.py": "",
            "config.py": "",
            "database.py": "",
            "model_loader.py": "",
        },
        "services": {
            "__init__.py": "",
            "prediction_service.py": "",
            "simulation_service.py": "",
            "standings_service.py": "",
            "toss_service.py": "",
        },
        "routers": {
            "__init__.py": "",
            "predict.py": "",
            "toss.py": "",
            "simulation.py": "",
            "standings.py": "",
        },
    }
}


def create_structure(base_path, structure_dict):
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)


if __name__ == "__main__":
    create_structure(".", structure)
    print("✅ API folder structure created successfully!")