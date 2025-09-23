import streamlit as st
import pandas as pd

# Optional AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False
    AgGrid = GridOptionsBuilder = GridUpdateMode = None

def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill--{kind}">{text}</span>'

def sanitize_df(df: pd.DataFrame):
    return df.reset_index(drop=True) if (df is not None and not df.empty) else df

def render_grid(df: pd.DataFrame, col_defs=None, height=360, row_height=36, download_name=None, grid_key=None):
    df = sanitize_df(df) if df is not None else pd.DataFrame()
    if not HAS_AGGRID or df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True, wrapText=False, autoHeight=False, minWidth=110)
        if col_defs:
            for c, opts in col_defs.items():
                gb.configure_column(
                    c,
                    header_name=opts.get("header_name", c),
                    width=opts.get("width"),
                    maxWidth=opts.get("maxWidth"),
                    minWidth=opts.get("minWidth", 90),
                    cellClass=opts.get("cellClass", "center-text"),
                    headerClass=opts.get("headerClass", ""),
                    wrapText=opts.get("wrapText", False),
                    autoHeight=opts.get("autoHeight", False),
                )
        gb.configure_grid_options(domLayout="normal", ensureDomOrder=True, rowHeight=row_height)
        AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="balham",
            height=height,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            key=grid_key or f"ag_{download_name or 'grid'}",
        )
    if download_name and not df.empty:
        st.download_button(
            f"Download {download_name}.csv",
            data=df.to_csv(index=False).encode(),
            file_name=f"{download_name}.csv",
            mime="text/csv",
            key=f"dl_{download_name}"
        )