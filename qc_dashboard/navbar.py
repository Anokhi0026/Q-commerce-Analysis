from streamlit_option_menu import option_menu
import streamlit as st

PAGES = {
    "Overview":           ("house",     "app.py"),
    "Objectives":         ("bullseye",  "pages/1_Objectives.py"),
    "Sampling & Design":  ("rulers",    "pages/2_Sampling.py"),
    "Questionnaire":      ("clipboard", "pages/3_Questionnaire.py"),
    "Demographics":       ("people",    "pages/4_Demographics.py"),
    "App Usage":          ("phone",     "pages/5_Obj1_Apps.py"),
    "Adoption":           ("link",      "pages/6_Obj2_Adoption.py"),
    "Behavior":           ("bar-chart", "pages/7_Obj3_Behavior.py"),
    "Drivers":            ("search",    "pages/8_Obj4_Drivers.py"),
    "Predictive":         ("robot",     "pages/9_Obj5_Predictive.py"),
    "Cluster Analysis":   ("puzzle",    "pages/11_Cluster_Analysis.py"),
    "Correspondence":     ("geo-alt",   "pages/12_Correspondence_Analysis.py"),
    "Summary":            ("stars",     "pages/10_Summary.py"),
}

def navbar():
    # Detect current page from session state
    current = st.session_state.get("current_page", "app.py")
    page_paths = [v[1] for v in PAGES.values()]
    page_names = list(PAGES.keys())
    
    # Find index of current page so it stays highlighted
    try:
        current_index = page_paths.index(current)
    except ValueError:
        current_index = 0

    selected = option_menu(
        menu_title=None,
        options=page_names,
        icons=[v[0] for v in PAGES.values()],
        default_index=current_index,  # ← highlights the actual current page
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

    # Only navigate if user clicked a DIFFERENT page
    selected_path = PAGES[selected][1]
    if selected_path != current:
        st.switch_page(selected_path)
