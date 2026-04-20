from streamlit_option_menu import option_menu
import streamlit as st

PAGES = {
    "Overview":              ("house",      "app"),
    "Objectives":            ("bullseye",   "pages/1_Objectives"),
    "Sampling & Design":     ("rulers",     "pages/2_Sampling"),
    "Questionnaire":         ("clipboard",  "pages/3_Questionnaire"),
    "Demographics":          ("people",     "pages/4_Demographics"),
    "App Usage":             ("phone",      "pages/5_Obj1_Apps"),
    "Adoption":              ("link",       "pages/6_Obj2_Adoption"),
    "Behavior":              ("bar-chart",  "pages/7_Obj3_Behavior"),
    "Drivers":               ("search",     "pages/8_Obj4_Drivers"),
    "Predictive":            ("robot",      "pages/9_Obj5_Predictive"),
    "Cluster Analysis":      ("puzzle",     "pages/11_Cluster_Analysis"),
    "Correspondence":        ("geo-alt",    "pages/12_Correspondence_Analysis"),
    "Summary":               ("stars",      "pages/10_Summary"),
}

def navbar():
    option_menu(
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
