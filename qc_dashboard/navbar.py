from streamlit_option_menu import option_menu
import streamlit as st
import os

# ── Page registry ──────────────────────────────────────────────────────────────
# Keys   = display name in navbar
# Values = (icon, path relative to THIS file's directory)
PAGES = {
    "Overview":           ("house",        "app.py"),
    "Objectives":         ("bullseye",     "pages/1_Objectives.py"),
    "Sampling & Design":  ("rulers",       "pages/2_Sampling.py"),
    "Questionnaire":      ("clipboard",    "pages/3_Questionnaire.py"),
    "Demographics":       ("people",       "pages/4_Demographics.py"),
    "App Usage":          ("phone",        "pages/5_Obj1_Apps.py"),
    "Adoption":           ("link",         "pages/6_Obj2_Adoption.py"),
    "Behavior":           ("bar-chart",    "pages/7_Obj3_Behavior.py"),
    "Correspondence":     ("geo-alt",      "pages/12_Correspondence_Analysis.py"),
    "Drivers":            ("search",       "pages/8_Obj4_Drivers.py"),
    "Cluster Analysis":   ("puzzle",       "pages/11_Cluster_Analysis.py"),
    "Predictive":         ("robot",        "pages/9_Obj5_Predictive.py"),
    "Non-User Analysis":  ("slash-circle", "pages/13_NonUser_Analysis.py"),
    "Summary":            ("stars",        "pages/10_Summary.py"),
}

# ── Resolve the app root directory once ───────────────────────────────────────
# navbar.py sits in the same folder as app.py, so __file__'s parent IS the root
_ROOT = os.path.dirname(os.path.abspath(__file__))

def _abs(rel_path: str) -> str:
    """Convert a relative page path to the absolute path st.switch_page needs."""
    return os.path.join(_ROOT, rel_path)

def navbar():
    # ── Detect current page from session state ─────────────────────────────────
    current_rel = st.session_state.get("current_page", "app.py")
    page_rels   = [v[1] for v in PAGES.values()]
    page_names  = list(PAGES.keys())

    try:
        current_index = page_rels.index(current_rel)
    except ValueError:
        current_index = 0

    # ── Render horizontal menu ─────────────────────────────────────────────────
    selected = option_menu(
        menu_title=None,
        options=page_names,
        icons=[v[0] for v in PAGES.values()],
        default_index=current_index,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0px 0px 8px 0px",
                "background-color": "#FAFAFA",
                "border-bottom": "1px solid #E2E8F0",
            },
            "nav-link": {
                "font-size": "12px",
                "padding": "6px 10px",
                "margin": "0px 1px",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#4F46E5",
                "color": "white",
                "font-weight": "600",
            },
        }
    )

    # ── Navigate only when user clicks a different page ────────────────────────
    selected_rel = PAGES[selected][1]
    if selected_rel != current_rel:
        st.switch_page(_abs(selected_rel))
