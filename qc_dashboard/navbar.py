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
    "Correspondence":     ("geo-alt",   "pages/12_Correspondence_Analysis.py"),
    "Drivers":            ("search",    "pages/8_Obj4_Drivers.py"),
    "Cluster Analysis":   ("puzzle",    "pages/11_Cluster_Analysis.py"),
    "Predictive":         ("robot",     "pages/9_Obj5_Predictive.py"),
    "Summary":            ("stars",     "pages/10_Summary.py"),
}

    def navbar():
    st.markdown("""
    <style>
    
    /* GLOBAL */
    html, body, [class*="css"] {
        font-size: 20px !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* 🔥 FIX HEADINGS (ADD THIS PART HERE) */
    [data-testid="stMarkdownContainer"] h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 34px !important;
    }
    
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 28px !important;
    }
    
    /* TEXT */
    p, label, span {
        font-size: 22px !important;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }
    
    /* BUTTONS */
    button {
        font-size: 18px !important;
    }
    
    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        font-size: 16px !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    # Detect current page from session state
    current = st.session_state.get("current_page", "app.py")
    page_paths = [v[1] for v in PAGES.values()]
    page_names = list(PAGES.keys())
    
    try:
        current_index = page_paths.index(current)
    except ValueError:
        current_index = 0

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
                "font-size": "16px",  # 👈 increased for visibility
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

    selected_path = PAGES[selected][1]
    if selected_path != current:
        st.switch_page(selected_path)
