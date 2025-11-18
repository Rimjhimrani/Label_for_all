import streamlit as st
import pandas as pd
import os
import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Page Configuration ---
st.set_page_config(
    page_title="Part Label Generator",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions for PDF Labels ---

# Styles for Single-Part Labels (Packing Factor = 1)
single_part_style = ParagraphStyle(
    name='Single_Part_Style',
    fontName='Helvetica-Bold',
    fontSize=10,
    alignment=TA_LEFT,
    leading=32,
    spaceBefore=0,
    spaceAfter=2,
    wordWrap='CJK'
)
single_desc_style = ParagraphStyle(
    name='Single_Desc_Style',
    fontName='Helvetica',
    fontSize=20,
    alignment=TA_LEFT,
    leading=16,
    spaceBefore=2,
    spaceAfter=2
)

# Styles for Multi-Part Labels (Packing Factor = 0.5)
multi_part_style = ParagraphStyle(
    name='Multi_Part_Style',
    fontName='Helvetica-Bold',
    fontSize=10,
    alignment=TA_LEFT,
    leading=20,
    spaceBefore=2,
    spaceAfter=2
)

# --- Formatting Functions ---

def format_single_part_no(part_no):
    """Formats a part number for a large, single-part label."""
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
        return Paragraph("<b><font size=34>EMPTY</font></b>", single_part_style)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", single_part_style)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", single_part_style)

def format_single_description(desc):
    """Formats a description for a large, single-part label."""
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, single_desc_style)

def format_multi_part_no(part_no):
    """Formats a part number for a smaller, multi-part label."""
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", multi_part_style)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", multi_part_style)

def format_multi_description(desc):
    """Formats a description for a multi-part label with dynamic font sizing."""
    if not desc or not isinstance(desc, str): desc = str(desc)
    
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 9
    desc = desc[:100] + "..." if len(desc) > 100 else desc

    dynamic_style = ParagraphStyle(
        name=f'Multi_Desc_Style_{font_size}', fontName='Helvetica', fontSize=font_size,
        alignment=TA_LEFT, leading=font_size + 2, spaceBefore=1, spaceAfter=1
    )
    return Paragraph(desc, dynamic_style)

# --- Core Logic Functions ---

def find_required_columns(df):
    """Finds essential columns in the dataframe, including Packing Factor."""
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    # --- CHANGE: Added Packing Factor lookup ---
    packing_factor_key = next((k for k in cols if 'PACKING' in k and 'FACTOR' in k), None)
    
    return (cols.get(part_no_key), cols.get(desc_key), cols.get(bus_model_key),
            cols.get(station_no_key), cols.get(container_type_key), cols.get(packing_factor_key))

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    """Assigns parts to physical rack locations based on Station, Container, and Packing Factor."""
    part_no_col, desc_col, model_col, station_col, container_col, packing_col = find_required_columns(df)
    if not all([part_no_col, container_col, station_col, packing_col]):
        st.error("❌ Critical column not found. Ensure 'Part Number', 'Container Type', 'Station No', and 'Packing Factor' columns exist.")
        return None

    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description', model_col: 'Bus Model',
        station_col: 'Station No', container_col: 'Container', packing_col: 'Packing Factor'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    df_processed['Packing Factor'] = pd.to_numeric(df_processed['Packing Factor'], errors='coerce')

    # --- CHANGE: 'bins_per_cell' is now determined by Packing Factor ---
    df_processed['bins_per_cell'] = df_processed['Packing Factor'].apply(lambda pf: 1 if pf == 1 else 2 if pf == 0.5 else 0)
    
    final_df_parts = []
    available_cells = []
    
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
        
        for level in sorted(config.get('levels', [])):
            for i in range(config.get('cells_per_level', 0)):
                location = {'Level': level, 'Physical_Cell': f"{i + 1:02d}", 'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd}
                available_cells.append(location)
    
    current_cell_index = 0
    last_processed_station = "N/A"

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no
        
        # --- CHANGE: Group also by Packing Factor to handle different capacities ---
        parts_grouped = station_group.groupby(['Container', 'Packing Factor'])
        
        for (container_type, packing_factor), group_df in parts_grouped:
            parts_to_assign = group_df.to_dict('records')
            bins_per_cell = parts_to_assign[0]['bins_per_cell'] if parts_to_assign else 0

            if bins_per_cell == 0: continue

            for i in range(0, len(parts_to_assign), bins_per_cell):
                if current_cell_index >= len(available_cells):
                    st.error(f"❌ Ran out of rack space at Station {station_no}.")
                    break
                
                chunk = parts_to_assign[i:i + bins_per_cell]
                current_location = available_cells[current_cell_index]
                
                for part in chunk:
                    part.update(current_location)
                    final_df_parts.append(part)
                
                current_cell_index += 1
            
            if current_cell_index >= len(available_cells): break
        if current_cell_index >= len(available_cells):
            st.warning("⚠️ All available rack space has been filled.")
            break
            
    for i in range(current_cell_index, len(available_cells)):
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_processed_station, 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)

    return pd.DataFrame(final_df_parts) if final_df_parts else pd.DataFrame()

def extract_location_values(row):
    """Extracts location values for the label."""
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]

# --- PDF Generation Function ---

def generate_combined_pdf(df, progress_bar=None, status_text=None):
    """Generates a single PDF with labels for both single and multiple parts per location."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    # Sort by physical location to ensure logical print order
    df.sort_values(by=['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell'], inplace=True, na_position='last')
    
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    # --- CHANGE: Group by physical location to handle single vs. multi-part labels ---
    location_cols = ['Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']
    grouped_by_location = df_parts_only.groupby(location_cols)
    
    total_locations = len(grouped_by_location)
    label_count = 0
    label_summary = {}

    for i, (loc_key, group) in enumerate(grouped_by_location):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Generating Label {i+1}/{total_locations}")
        
        rack_num = f"{group.iloc[0].get('Rack No 1st', '0')}{group.iloc[0].get('Rack No 2nd', '0')}"
        rack_key = f"Rack {rack_num.zfill(2)}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if label_count > 0 and label_count % 4 == 0:
            elements.append(PageBreak())

        # Common location values and table setup
        location_values = extract_location_values(group.iloc[0])
        location_headers = ['Bus Model', 'Station No', 'Rack', 'R1', 'R2', 'Level', 'Cell']
        location_data = [location_headers, location_values] # Will use only values later
        
        col_widths = [2.9, 1.3, 1.2, 1.3, 1.3, 1.3] # Proportions for location cells
        location_widths = [4 * cm] + [w * (15 * cm - 4*cm) / sum(col_widths) for w in col_widths]
        
        location_table = Table([['Line Location'] + location_values], colWidths=location_widths, rowHeights=1.2 * cm)
        
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, 0), 16), ('FONTSIZE', (1, 0), (-1, -1), 16)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        
        # --- LOGIC FOR SINGLE vs MULTI PART ---
        if len(group) == 1: # Single Part Label (Packing Factor 1)
            part = group.iloc[0]
            part_table = Table([['Part No', format_single_part_no(str(part['Part No']))], ['Description', format_single_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
            part_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (1, 0), (-1, -1), 5),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)
            ]))
            elements.append(part_table)

        elif len(group) >= 2: # Multi-Part Label (Packing Factor 0.5)
            part1 = group.iloc[0]
            part2 = group.iloc[1]
            part_table1 = Table([['Part No', format_multi_part_no(str(part1['Part No']))], ['Description', format_multi_description(str(part1['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            part_table2 = Table([['Part No', format_multi_part_no(str(part2['Part No']))], ['Description', format_multi_description(str(part2['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            
            style = TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('LEFTPADDING', (1, 0), (-1, -1), 5),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (0, -1), 16)
            ])
            part_table1.setStyle(style)
            part_table2.setStyle(style)

            elements.append(part_table1)
            elements.append(Spacer(1, 0.1 * cm))
            elements.append(part_table2)

        # Add shared location table and spacing
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(location_table)
        elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

# --- Main Application UI ---
def main():
    st.title("🏷️ Rack Label Generator")
    st.markdown("<p style='font-style:italic;'>Designed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    base_rack_id = st.sidebar.text_input("Storage Line Side ID", "R")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'], help="Your file must contain 'Part Number', 'Description', 'Station No', 'Container Type', and 'Packing Factor' columns.")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            _, _, _, _, _, packing_col = find_required_columns(df)
            
            if packing_col:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Global Rack Configuration")
                
                levels = st.sidebar.multiselect("Active Levels (for all racks)", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells_per_level = st.sidebar.number_input("Number of Physical Cells per Level", min_value=1, value=10, step=1)
                num_racks = st.sidebar.number_input("Total Number of Racks", min_value=1, value=4, step=1)
                
                if st.button("🚀 Generate PDF Labels", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        # --- CHANGE: Filter out Packing Factor 0.0625 ---
                        df[packing_col] = pd.to_numeric(df[packing_col], errors='coerce')
                        df_to_process = df[df[packing_col].isin([1, 0.5])].copy()
                        ignored_count = len(df) - len(df_to_process)
                        if ignored_count > 0:
                            st.info(f"ℹ️ Ignored {ignored_count} rows where Packing Factor was not 1 or 0.5.")

                        rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': num_cells_per_level} for i in range(num_racks)}

                        df_assigned = automate_location_assignment(df_to_process, base_rack_id, rack_configs, status_text)
                        
                        if df_assigned is not None and not df_assigned.empty:
                            pdf_buffer, label_summary = generate_combined_pdf(df_assigned, progress_bar, status_text)
                            
                            if pdf_buffer:
                                total_labels = sum(label_summary.values())
                                status_text.text(f"✅ PDF with {total_labels} labels generated successfully!")
                                file_name = f"{os.path.splitext(uploaded_file.name)[0]}_labels.pdf"
                                st.download_button(label="📥 Download PDF", data=pdf_buffer.getvalue(), file_name=file_name, mime="application/pdf")

                                if total_labels > 0:
                                    st.markdown("---")
                                    st.subheader("📊 Generation Summary")
                                    summary_df = pd.DataFrame(list(label_summary.items()), columns=['Rack', 'Labels Generated'])
                                    st.table(summary_df.sort_values(by='Rack').reset_index(drop=True))
                        else:
                            st.error("❌ No data was processed. Check your input file and configurations.")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {e}")
                        st.exception(e)
                    finally:
                        progress_bar.empty()
                        status_text.empty()
            else:
                st.error("❌ A 'Packing Factor' column could not be found in the uploaded file. This column is required for automation.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a file to begin.")

if __name__ == "__main__":
    main()
