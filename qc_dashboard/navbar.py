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
    selected = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=[v[0] for v in PAGES.values()],
        default_index=0,
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

    # Navigate to selected page
    if selected and PAGES[selected][1] != st.session_state.get("current_page"):
        st.switch_page(PAGES[selected][1])
