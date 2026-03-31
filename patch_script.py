import re
import sys

ADAPTIVE_BLOCK = '''\n            # ADAPTIVE FIREPLACE LEARNING INTEGRATION\n            fireplace_power_kw = None\n            if fireplace_on:\n                other_rooms_temp = avg_other_rooms_temp\n                from .model_wrapper import get_enhanced_model_wrapper\n                wrapper = get_enhanced_model_wrapper()\n                fireplace_analysis = (\n                    wrapper.adaptive_fireplace\n                    ._calculate_learned_heat_contribution(\n                        temp_differential=actual_indoor - other_rooms_temp,\n                        outdoor_temp=outdoor_temp,\n                        fireplace_active=True,\n                    )\n                )\n                fireplace_power_kw = fireplace_analysis.get(\n                    "heat_contribution_kw", 0.0\n                )\n                logging.debug(\n                    f"Adaptive fireplace learning: power={fireplace_power_kw:.2f} kW"\n                )\n'''

def patch_tolerance(model_wrapper_path):
    with open(model_wrapper_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if tolerance is already set to 0.05
    if re.search(r'tolerance\s*=\s*0\.05\s*#\s*°C', content):
        print(f"Tolerance already set to 0.05 in {model_wrapper_path}. No changes made.")
        return

    # Replace tolerance value if found
    patched = re.sub(r'(tolerance\s*=\s*)0\.1(\s*#\s*°C)', r'\g<1>0.05\2', content)

    with open(model_wrapper_path, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f"Patched {model_wrapper_path}.")


def patch_smart_rounding(content):
    """
    Replace smart_rounded_temp with final_temp rounded to one decimal place in ha_client.set_state.
    Returns patched content.
    """
    set_state_pattern = re.compile(
        r'(ha_client\.set_state\(\s*config\.TARGET_OUTLET_TEMP_ENTITY_ID,\s*)smart_rounded_temp(,)',
        re.DOTALL
    )
    def replace_set_state(match):
        return f'{match.group(1)}round(final_temp, 1){match.group(2)}'
    patched_content = set_state_pattern.sub(replace_set_state, content)
    if patched_content == content:
        print("No smart_rounded_temp usages found to replace in ha_client.set_state.")
    else:
        print("Smart_rounded_temp replaced with final_temp in ha_client.set_state.")
    return patched_content


def patch_applied_temp(content):
    """
    Replace smart_rounded_temp with final_temp in applied_temp assignment.
    Handles leading whitespace, parentheses, and line breaks.
    Returns patched content.
    """
    applied_temp_pattern = re.compile(
        r'(^[ \t]*applied_temp\s*=\s*\(\s*)smart_rounded_temp(\s*if\s*not\s*config\.SHADOW_MODE\s*else\s*final_temp\s*\))',
        re.DOTALL | re.MULTILINE
    )
    def replace_applied_temp(match):
        return f'{match.group(1)}final_temp{match.group(2)}'
    patched_content = applied_temp_pattern.sub(replace_applied_temp, content)
    if patched_content == content:
        print("No smart_rounded_temp usages found to replace in applied_temp assignment.")
    else:
        print("Smart_rounded_temp replaced with final_temp in applied_temp assignment.")
    return patched_content


def patch_fireplace_learning(content):
    """
    Insert adaptive fireplace learning block after feature engineering.
    Returns patched content.
    """
    if 'ADAPTIVE FIREPLACE LEARNING INTEGRATION' in content:
        print("Adaptive fireplace block already present. No changes made.")
        return content
    pattern = re.compile(r'(\n\s*# --- Step 2: Feature Building ---.*?\n)', re.DOTALL)
    match = pattern.search(content)
    if not match:
        print("Feature engineering block not found.")
        return content
    start = match.end()
    patched = content[:start] + ADAPTIVE_BLOCK + content[start:]
    print("Adaptive fireplace learning block inserted.")
    return patched


def patch_main_py(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = patch_smart_rounding(content)
    content = patch_applied_temp(content)
    content = patch_fireplace_learning(content)

    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched {main_path}.")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main_path = sys.argv[1]
        model_wrapper_path = sys.argv[2]
    else:
        print("No arguments provided. Using default src/main.py and src/model_wrapper.py.")
        main_path = 'src/main.py'
        model_wrapper_path = 'src/model_wrapper.py'
    patch_main_py(main_path)
    patch_tolerance(model_wrapper_path)
