# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate VS Code settings for the InsertAnything project.

This script merges the python.analysis.extraPaths from the "{ISAACSIM_DIR}/.vscode/settings.json" file into
the ".vscode/settings.json" file.

The script reads Isaac Sim's Python analysis paths and adds the local extension source directory.
"""

import argparse
import os
import pathlib
import re
import sys

PROJECT_DIR = pathlib.Path(__file__).parents[2]
"""Path to the project directory."""

parser = argparse.ArgumentParser(description="Set up VS Code for InsertAnything.")
parser.add_argument("--isaac_path", type=str, default=os.environ.get("ISAAC_PATH"), help="Isaac Sim directory.")
args = parser.parse_args()
isaacsim_dir = args.isaac_path

# check if the isaac-sim directory exists
if not isaacsim_dir or not os.path.exists(isaacsim_dir):
    raise FileNotFoundError(
        f"Could not find the Isaac Sim directory: {isaacsim_dir}. Pass it with --isaac_path or set ISAAC_PATH."
    )

ISAACSIM_DIR = isaacsim_dir
"""Path to the isaac-sim directory."""


def overwrite_python_analysis_extra_paths(isaaclab_settings: str) -> str:
    """Overwrite the python.analysis.extraPaths in the Isaac Lab settings file.

    The extraPaths are replaced with the path names from the isaac-sim settings file that exists in the
    "{ISAACSIM_DIR}/.vscode/settings.json" file.

    If the isaac-sim settings file does not exist, the extraPaths are not overwritten.

    Args:
        isaaclab_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten python analysis extra paths.
    """
    # isaac-sim settings
    isaacsim_vscode_filename = os.path.join(ISAACSIM_DIR, ".vscode", "settings.json")

    # we use the isaac-sim settings file to get the python.analysis.extraPaths for kit extensions
    # if this file does not exist, we will not add any extra paths
    if os.path.exists(isaacsim_vscode_filename):
        # read the path names from the isaac-sim settings file
        with open(isaacsim_vscode_filename) as f:
            vscode_settings = f.read()
        # extract the path names
        # search for the python.analysis.extraPaths section and extract the contents
        settings = re.search(
            r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
        )
        settings = settings.group(0)
        settings = settings.split('"python.analysis.extraPaths": [')[-1]
        settings = settings.split("]")[0]

        # read the path names from the isaac-sim settings file
        path_names = settings.split(",")
        path_names = [path_name.strip().strip('"') for path_name in path_names]
        path_names = [path_name for path_name in path_names if len(path_name) > 0]

        # change the path names to be relative to the Isaac Lab directory
        rel_path = os.path.relpath(ISAACSIM_DIR, PROJECT_DIR)
        path_names = ['"${workspaceFolder}/' + rel_path + "/" + path_name + '"' for path_name in path_names]
    else:
        path_names = []
        print(
            f"[WARN] Could not find Isaac Sim VSCode settings: {isaacsim_vscode_filename}."
            "\n\tThis will result in missing 'python.analysis.extraPaths' in the VSCode"
            "\n\tsettings, which limits the functionality of the Python language server."
            "\n\tHowever, it does not affect the functionality of the Isaac Lab project."
            "\n\tWe are working on a fix for this issue with the Isaac Sim team."
        )

    # add the path names that are in the Isaac Lab extensions directory
    isaaclab_extensions = os.listdir(os.path.join(PROJECT_DIR, "source"))
    path_names.extend(['"${workspaceFolder}/source/' + ext + '"' for ext in isaaclab_extensions])

    # combine them into a single string
    path_names = ",\n\t\t".expandtabs(4).join(path_names)
    path_names = path_names.replace("\\", "/")

    # replace the path names in the Isaac Lab settings file with the path names parsed
    isaaclab_settings = re.sub(
        r"\"python.analysis.extraPaths\": \[.*?\]",
        '"python.analysis.extraPaths": [\n\t\t'.expandtabs(4) + path_names + "\n\t]".expandtabs(4),
        isaaclab_settings,
        flags=re.DOTALL,
    )
    # return the Isaac Lab settings string
    return isaaclab_settings


def overwrite_default_python_interpreter(isaaclab_settings: str) -> str:
    """Overwrite the default python interpreter in the Isaac Lab settings file.

    The default python interpreter is replaced with the path to the python interpreter used by the
    isaac-sim project. This is necessary because the default python interpreter is the one shipped with
    isaac-sim.

    Args:
        isaaclab_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten default python interpreter.
    """
    # read executable name
    python_exe = os.path.normpath(sys.executable)

    # Replace the bundled interpreter with Isaac Sim's Linux launcher.
    if f"kit{os.sep}python{os.sep}bin{os.sep}python" in python_exe:
        python_exe = python_exe.replace(f"kit{os.sep}python{os.sep}bin{os.sep}python3", "python.sh")

    # replace the default python interpreter in the Isaac Lab settings file with the path to the
    # python interpreter in the Isaac Lab directory
    isaaclab_settings = re.sub(
        r"\"python.defaultInterpreterPath\": \".*?\"",
        f'"python.defaultInterpreterPath": "{python_exe}"',
        isaaclab_settings,
        flags=re.DOTALL,
    )
    # return the Isaac Lab settings file
    return isaaclab_settings


def main():
    # Isaac Lab template settings
    isaaclab_vscode_template_filename = os.path.join(PROJECT_DIR, ".vscode", "tools", "settings.template.json")
    # make sure the Isaac Lab template settings file exists
    if not os.path.exists(isaaclab_vscode_template_filename):
        raise FileNotFoundError(
            f"Could not find the Isaac Lab template settings file: {isaaclab_vscode_template_filename}"
        )
    # read the Isaac Lab template settings file
    with open(isaaclab_vscode_template_filename) as f:
        isaaclab_template_settings = f.read()

    # overwrite the python.analysis.extraPaths in the Isaac Lab settings file with the path names
    isaaclab_settings = overwrite_python_analysis_extra_paths(isaaclab_template_settings)
    # overwrite the default python interpreter in the Isaac Lab settings file with the path to the
    # python interpreter used to call this script
    isaaclab_settings = overwrite_default_python_interpreter(isaaclab_settings)

    # add template notice to the top of the file
    header_message = (
        "// This file is a template and is automatically generated by the setup_vscode.py script.\n"
        "// Do not edit this file directly.\n"
        "// \n"
        f"// Generated from: {isaaclab_vscode_template_filename}\n"
    )
    isaaclab_settings = header_message + isaaclab_settings

    # write the Isaac Lab settings file
    isaaclab_vscode_filename = os.path.join(PROJECT_DIR, ".vscode", "settings.json")
    with open(isaaclab_vscode_filename, "w") as f:
        f.write(isaaclab_settings)

    # copy the launch.json file if it doesn't exist
    isaaclab_vscode_launch_filename = os.path.join(PROJECT_DIR, ".vscode", "launch.json")
    isaaclab_vscode_template_launch_filename = os.path.join(PROJECT_DIR, ".vscode", "tools", "launch.template.json")
    if not os.path.exists(isaaclab_vscode_launch_filename):
        # read template launch settings
        with open(isaaclab_vscode_template_launch_filename) as f:
            isaaclab_template_launch_settings = f.read()
        # add header
        header_message = header_message.replace(
            isaaclab_vscode_template_filename, isaaclab_vscode_template_launch_filename
        )
        isaaclab_launch_settings = header_message + isaaclab_template_launch_settings
        # write the Isaac Lab launch settings file
        with open(isaaclab_vscode_launch_filename, "w") as f:
            f.write(isaaclab_launch_settings)


if __name__ == "__main__":
    main()
